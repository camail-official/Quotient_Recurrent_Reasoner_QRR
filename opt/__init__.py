"""QRR inference-only optimization stack (flag-gated, default OFF).

Sudoku-shape z-step 48.4 -> ~34 ms on A100-40 (see the paper's runtime appendix). Six
pieces, all INFERENCE-ONLY -- none has a backward, so this module must never be active in
a training process. Training launchers never set the flag; on top of that every
wrapper falls back to the default path whenever gradients are enabled.

  K1   fused gauge-conv2d (triton)             op parity 5.1e-08
  K3'  cached input-routing extras + fused dress   math identical
  K2   fused re_defect (triton)                op parity 4.1e-07 vs the eager re_last
  lazy skip per-step decode_logprobs           harness decodes at fire/cap explicitly
  S2   reset_carry identity when nothing resets  bitwise
  bf16 dressed-SDPA input dtype                not bitwise: check accuracy per checkpoint

Enable with opt.install() after importing models (evaluate.py --opt); each piece can be
turned off by keyword. Class-level installation: harnesses that rebuild nets keep the stack.

NOTE `lazy` returns a placeholder for outputs["logits"]; use it only under harnesses
that decode explicitly (the released eval protocols do). It is skipped automatically
when the model is in training mode.
"""
import inspect
import textwrap
import weakref

import torch

_INSTALLED = set()


def _ver(t):
    """In-place version counter, or None for inference tensors (they carry none)."""
    return None if t.is_inference() else t._version


# Embedding cache for eval rollouts that recompute input embeddings on every z-step.
# Content-checked (torch.equal on the integer inputs) and keyed on the embedding weights'
# versions, so a new batch or an optimizer step always recomputes. Returning the SAME
# tensor object across z-steps is what lets the attention-extras cache below hit.
_EMB = weakref.WeakKeyDictionary()


def _emb_cached(inner, batch):
    inp, pid = batch["inputs"], batch["puzzle_identifiers"]
    if inner.training or torch.is_grad_enabled():
        return inner._input_embeddings(inp, pid)
    ws = [inner.embed_tokens.embedding_weight]
    if getattr(inner, "puzzle_emb", None) is not None:
        ws.append(inner.puzzle_emb.weights)
    wv = tuple(_ver(w) for w in ws)
    c = _EMB.get(inner)
    if (c is not None and c["wv"] == wv and c["inp"].shape == inp.shape
            and c["pid"].shape == pid.shape
            and torch.equal(c["inp"], inp) and torch.equal(c["pid"], pid)):
        return c["emb"]
    emb = inner._input_embeddings(inp, pid)
    _EMB[inner] = dict(inp=inp.clone(), pid=pid.clone(), wv=wv, emb=emb)
    return emb


# ---------------------------------------------------------------- K1: fused conv2d
def _install_conv():
    import models.gauge_layers as GL
    from opt.kernels import gauge_conv2d_fused
    orig = GL.GaugeConv.forward

    def conv_fwd(self, hc_grid, g_tok, mask_grid=None):
        if self.type == "conv1d" or g_tok is None or torch.is_grad_enabled():
            return orig(self, hc_grid, g_tok, mask_grid)
        return gauge_conv2d_fused(self, hc_grid, g_tok, mask_grid)

    GL.GaugeConv.forward = conv_fwd


# ---------------------------------------------------------------- K3': attn extras
def _wide_dress(qkvR, phR, phI, gR, gI, qe, ke, to_bf16: bool):
    a, b = qkvR[..., 0], qkvR[..., 1]
    ph_r = phR[None, :, None, :]
    ph_i = phI[None, :, None, :]
    g_r = gR[:, :, None, None]
    g_i = gI[:, :, None, None]
    wqk_r = ph_r * g_r + ph_i * g_i
    wqk_i = ph_i * g_r - ph_r * g_i
    wv_r, wv_i = g_r.expand_as(wqk_r), (-g_i).expand_as(wqk_i)
    wr = torch.stack([wqk_r, wqk_r, wv_r], dim=2)
    wi = torch.stack([wqk_i, wqk_i, wv_i], dim=2)
    o_r = a * wr - b * wi
    o_i = a * wi + b * wr
    out = torch.stack([o_r, o_i], dim=-1)
    out = out.reshape(*out.shape[:-2], out.shape[-2] * 2)
    if to_bf16:
        out = out.to(torch.bfloat16)
    out = out.permute(0, 2, 3, 1, 4)
    qh, kh, vh = out[:, 0], out[:, 1], out[:, 2]
    qh = torch.cat([qh, qe], -1)          # cat folds into the same inductor kernel
    kh = torch.cat([kh, ke], -1)
    return qh.contiguous(), kh.contiguous(), vh.contiguous()


def _install_attn():
    import models.gauge_layers as GL
    wide_c = torch.compile(_wide_dress, dynamic=False)
    orig = GL.GaugeAttention.forward
    # Per-module cache. The entry holds a STRONG reference to the xc it was built from, so
    # no other tensor can occupy that storage while cached: a data_ptr match then means the
    # same storage (same content unless mutated in place, which the version check covers
    # for ordinary tensors). Keyed by the module object (weak), never by id(): ids of freed
    # modules are reused across net rebuilds. Routing-weight versions invalidate on updates.
    extras = weakref.WeakKeyDictionary()

    def attn_fwd(self, hc, Uri=None, xc=None):
        if (getattr(self, "xdim", 0) <= 0 or Uri is None or len(Uri) != 4
                or Uri[3] is None or xc is None or torch.is_grad_enabled()):
            return orig(self, hc, Uri=Uri, xc=xc)
        B, S, C = hc.shape
        qkv = self.qkv(hc).reshape(B, S, 3, self.h, self.dc)
        g = Uri[3]
        am = None if Uri[2] is None else Uri[2].unsqueeze(1)
        ph = self.rope.phase[:S]
        key = (xc.data_ptr(), tuple(xc.shape), _ver(xc), hc.dtype,
               _ver(self.x_q), _ver(self.x_k), _ver(self.x_rel))
        ent = extras.get(self)
        if ent is None or ent[0] != key:
            qe, ke, xm = self._x_sdpa_extras(
                xc, torch.bfloat16 if GL.DRESSED_BF16 else torch.float32)
            ent = (key, (qe.contiguous(), ke.contiguous(), xm), xc)
            extras[self] = ent
        qe, ke, xm = ent[1]
        qh, kh, vh = wide_c(torch.view_as_real(qkv), ph.real.contiguous(),
                            ph.imag.contiguous(), g.real.contiguous(),
                            g.imag.contiguous(), qe, ke, GL.DRESSED_BF16)
        am = xm if am is None else am + xm
        o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, attn_mask=am,
                                                             scale=self.scale)
        fn2 = GL._get_compiled("dress_post", GL._dress_post)
        out = torch.view_as_complex(fn2(o, g.real.contiguous(), g.imag.contiguous()))
        return self.o(out.reshape(B, S, self.h * self.dc))

    GL.GaugeAttention.forward = attn_fwd


# ------------------------------------------------- K2 + lazy: forward source transform
def _install_redefect_lazy(redefect: bool, lazy: bool):
    import models.qrr_model as QM
    from opt.kernels import redefect_fused
    cls = QM.QRRModelInner
    src = textwrap.dedent(inspect.getsource(cls.forward))
    A = '_y1 = z_state["y"].detach()'
    B_ = 'z_state["re_last"] = _dre'
    ia, ib = src.find(A), src.find(B_)
    assert ia != -1 and ib != -1 and ib > ia, "re_defect anchors not found -- source changed"
    line_start = src.rfind("\n", 0, ia) + 1
    pad = src[line_start:ia]
    repl = (f'_dre = __REDEFECT__(_re_py, _re_pm if "m" in z_state else None,\n'
            f'{pad}                  z_state["y"].detach(),\n'
            f'{pad}                  z_state["m"].detach() if "m" in z_state else None)\n'
            f'{pad}z_state["re_last"] = _dre')
    if redefect:
        src = src[:ia] + repl + src[ib + len(B_):]
    E = 'input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])'
    assert src.count(E) == 1, "embedding anchor not found -- source changed"
    src = src.replace(E, 'input_embeddings = __EMB__(self, batch)')
    L = 'output = self.decode_logprobs(z_state["y"], input_embeddings, batch["inputs"])'
    il = src.find(L)
    assert il != -1, "decode anchor not found -- source changed"
    src = (src[:il] + 'output = (z_state["y"][:, :1, :1] * 0 '
           'if (__LAZY_LM__ and not self.training) else '
           'self.decode_logprobs(z_state["y"], input_embeddings, batch["inputs"]))'
           + src[il + len(L):])
    g = dict(vars(QM))
    g["__REDEFECT__"] = redefect_fused
    g["__LAZY_LM__"] = bool(lazy)
    g["__EMB__"] = _emb_cached
    exec(compile(src, "<opt_forward>", "exec"), g)
    cls.forward = g["forward"]


# ---------------------------------------------------------------- S2: reset identity
def _install_reset():
    import models.qrr_model as QM
    cls = QM.QRRModelInner
    orig = cls.reset_carry

    def rc_fast(self, reset_flag, batch, carry):
        if carry is None or carry.z_L_state is None or bool(reset_flag.any()):
            return orig(self, reset_flag, batch, carry)
        return carry

    cls.reset_carry = rc_fast


def install(conv=True, attn=True, redefect=True, lazy=True, reset=True, bf16=True):
    """Class-level install; idempotent per piece. Call AFTER importing models."""
    import models.gauge_layers as GL
    done = []
    if conv and "conv" not in _INSTALLED:
        _install_conv(); _INSTALLED.add("conv"); done.append("K1")
    if attn and "attn" not in _INSTALLED:
        _install_attn(); _INSTALLED.add("attn"); done.append("K3'")
    if (redefect or lazy) and "fwd" not in _INSTALLED:
        _install_redefect_lazy(redefect=redefect, lazy=lazy); _INSTALLED.add("fwd")
        done.append("+".join((["K2"] if redefect else []) + (["lazy"] if lazy else [])))
    if reset and "reset" not in _INSTALLED:
        _install_reset(); _INSTALLED.add("reset"); done.append("S2")
    if bf16 and "bf16" not in _INSTALLED:
        GL.DRESSED_BF16 = True; _INSTALLED.add("bf16"); done.append("bf16-SDPA")
    print(f"[opt] inference stack installed: {'+'.join(done) if done else '(already)'}")
    return done

