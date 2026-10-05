"""Name-seeded deferred parameter init.

Module init normally draws from ONE global torch stream (pretrain.py seeds it with
`config.seed + RANK`) in CONSTRUCTION ORDER, so inserting a module changes the init of
every module built after it.  Parameters created with `new_param` avoid this:

    from models.rng_streams import new_param
    self.w = new_param(out, in, init=lambda p, g: p.normal_(0, 0.02, generator=g),
                       name="block0.w")            # torch.empty -> draws NO global RNG

and the builder calls `init_deferred(seed)` ONCE after the model is constructed
(create_model.py does this).  Other modules then see exactly the draws they would have
seen without the deferred ones.

Each deferred parameter is seeded from its NAME, not its position, so registration order
does not matter either.  In this release the input-routing factors (xroute.*) and the
SwiGLU gate anchors (gate.*) are deferred.
"""
import torch

_DEFERRED = []


def _name_hash(name: str) -> int:
    """FNV-1a. Python's hash() is salted per process and would not reproduce."""
    h = 1469598103934665603
    for ch in name.encode():
        h = ((h ^ ch) * 1099511628211) % (2 ** 64)
    return h


def new_param(*shape, init, name, dtype=None, device=None, requires_grad=True):
    """A Parameter allocated with torch.empty -- consumes NO draw from the global stream.

    `init(param, generator)` runs later, inside init_deferred, under no_grad.
    """
    p = torch.nn.Parameter(torch.empty(*shape, dtype=dtype, device=device),
                           requires_grad=requires_grad)
    defer_init(p, init=init, name=name)
    return p


def defer_init(param, init, name):
    """Register an already-allocated tensor for deferred initialization."""
    if any(n == name for n, _, _ in _DEFERRED):
        raise ValueError(f"duplicate deferred-init name {name!r}: names seed the stream, "
                         "so they must be unique")
    _DEFERRED.append((name, param, init))


def reset_deferred():
    """Drop any pending registrations WITHOUT initializing them.

    init_deferred() is what normally drains the registry, but an EVAL-time build never calls
    it -- it constructs the model and immediately overwrites every parameter via
    load_state_dict. So a second build in the same process would hit the duplicate-name guard
    on entries left by the first. Those entries are stale by construction, so dropping them is
    correct; training still goes through init_deferred and is unaffected.
    """
    n = len(_DEFERRED)
    _DEFERRED.clear()
    return n


def pending():
    return [n for n, _, _ in _DEFERRED]


def init_deferred(seed: int):
    """Initialize every registered parameter from its own name-derived stream.

    Returns the list of names initialized (empty if nothing is registered).
    """
    names = []
    for name, p, init in _DEFERRED:
        g = torch.Generator(device=p.device)
        g.manual_seed((seed * 1000003 + _name_hash(name)) % (2 ** 63 - 1))
        with torch.no_grad():
            init(p, g)
        names.append(name)
    _DEFERRED.clear()
    return names
