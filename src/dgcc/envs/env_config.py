"""Fail-closed resolution of the `sim` config block into DLOLabEnv kwargs.

Motivation (reselection preflight, benchmark §0.1 / §5(c) items 0a-0c).
------------------------------------------------------------------------
Every launcher used to build the adapter kwargs by *explicitly enumerating*
a literal dict::

    return {"dt": float(sim.get("dt", 1e-3)), ..., "grasp_realism": ...}

That shape has three structural defects, all of which fail *open*:

1. **Silent drop.** A `sim` key the literal does not enumerate (a new
   parameter, or a typo such as ``move_vmax``) is discarded without a
   warning.  The adapter then runs on its own constructor defaults.
2. **Silent legacy fallback.** ``DLOLabEnv.__init__`` treats a missing
   ``move_v_max`` as "run the DEPRECATED pre-correction primitive", so a
   dropped key downgrades the physics instead of erroring.
3. **Silent default injection.** ``sim.get(key, default)`` means a config
   that never mentions a parameter still gets a hard-coded number that is
   invisible in the config SHA.

Combined, a governed launch could consume a config whose SHA is pinned by
the pre-registration and still train on the uncorrected environment with no
log line, no warning and no exception.  This module removes that class of
failure by construction:

* the accepted key set is **derived from the adapter signature itself**
  (``inspect.signature``), so it can never fall behind the adapter;
* an **unknown** ``sim`` key is a hard error;
* a **legacy** key (``move_step_size`` / ``move_hold_steps``) is a hard
  error in corrected mode;
* the corrected-mode **required** keys must be present *explicitly* — no
  defaulting;
* the fully resolved kwargs are returned for verbatim logging, and
  :func:`assert_corrected_env` re-verifies the *constructed* adapter so the
  guarantee survives even if a caller bypasses this resolver.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

__all__ = [
    "EnvConfigError",
    "CORRECTED_REQUIRED_KEYS",
    "LEGACY_SIM_KEYS",
    "allowed_sim_keys",
    "resolve_env_kwargs",
    "assert_corrected_env",
    "format_effective_env_params",
]


class EnvConfigError(RuntimeError):
    """Raised fail-closed for any unresolvable/ambiguous `sim` configuration.

    The message always starts with ``"env config"`` so log scanners can
    census it separately from the covenant error kinds.
    """


#: Keys whose presence *defines* the corrected (quasi-static) primitive.
#: ``DLOLabEnv`` activates the R1-R5 bundle iff ``move_v_max`` is not None,
#: and ``move_hold_max_steps`` is the R2 bound; requiring both explicitly
#: means a corrected launch cannot inherit either from a code default.
CORRECTED_REQUIRED_KEYS = ("move_v_max", "move_hold_max_steps")

#: Pre-correction parameters.  Retained in the adapter for one release
#: (its docstring says so) but forbidden in a corrected launch.
LEGACY_SIM_KEYS = ("move_step_size", "move_hold_steps")

#: `sim` never carries these -- they come from the `run` block / call site.
_NON_SIM_PARAMS = frozenset({"n_envs"})


def _env_signature(env_cls: type) -> inspect.Signature:
    return inspect.signature(env_cls.__init__)


def allowed_sim_keys(env_cls: type) -> tuple[str, ...]:
    """Adapter-derived whitelist of `sim` keys, in signature order."""
    params = _env_signature(env_cls).parameters
    return tuple(
        name
        for name, param in params.items()
        if name not in ("self", *_NON_SIM_PARAMS)
        and param.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


def _coercer(annotation: Any) -> Callable[[Any], Any]:
    """Map a (string, because of `from __future__ import annotations`)
    annotation onto the coercion the literal dicts used to apply inline."""
    text = str(annotation).replace(" ", "")
    text = text.replace("|None", "").replace("Optional[", "").rstrip("]")
    if text == "bool":
        return _coerce_bool
    if text == "int":
        return _coerce_int
    if text == "float":
        return float
    return lambda value: value


def _coerce_bool(value: Any) -> bool:
    # YAML already yields real booleans; anything else is a config mistake
    # worth surfacing rather than silently truthy-casting (`"false"` is True).
    if isinstance(value, bool):
        return value
    raise EnvConfigError(
        f"env config: expected a YAML boolean, got {value!r} ({type(value).__name__})"
    )


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnvConfigError(
            f"env config: expected an integer, got {value!r} ({type(value).__name__})"
        )
    if int(value) != value:
        raise EnvConfigError(f"env config: expected an integer, got {value!r}")
    return int(value)


def resolve_env_kwargs(
    config: dict[str, Any],
    n_envs: int,
    *,
    env_cls: type,
    require_corrected: bool,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve `config["sim"]` into explicit ``env_cls`` kwargs, fail-closed.

    Parameters
    ----------
    require_corrected:
        When True (the governed training path) the resolver refuses any
        configuration that would produce a pre-correction adapter: legacy
        keys are rejected, and every key in :data:`CORRECTED_REQUIRED_KEYS`
        must be present explicitly.  When False the resolver still rejects
        unknown keys but tolerates a legacy `sim` block, so diagnostic and
        historical-reproduction scripts keep working.
    overrides:
        Call-site values that are not config-driven (e.g. an oracle probe
        forcing ``grasp_realism=False``).  Applied after resolution and
        validated against the same whitelist, so an override cannot smuggle
        an unknown kwarg either.

    Returns the kwargs dict.  Keys absent from `sim` are simply omitted, so
    the adapter's own signature default applies and stays the single source
    of truth (the old `sim.get(key, literal)` duplicated -- and drifted from
    -- those defaults).
    """
    if "sim" not in config:
        raise EnvConfigError("env config: config has no `sim` block")
    sim = config["sim"]
    if not isinstance(sim, dict):
        raise EnvConfigError(f"env config: `sim` must be a mapping, got {type(sim).__name__}")

    whitelist = allowed_sim_keys(env_cls)
    unknown = [key for key in sim if key not in whitelist]
    if unknown:
        raise EnvConfigError(
            f"env config: unknown `sim` key(s) {sorted(unknown)}; "
            f"{env_cls.__name__} accepts {sorted(whitelist)}. "
            "Unknown keys are rejected instead of silently dropped -- a dropped "
            "key used to downgrade the run to the pre-correction adapter."
        )

    if require_corrected:
        legacy = [key for key in LEGACY_SIM_KEYS if key in sim]
        if legacy:
            raise EnvConfigError(
                f"env config: DEPRECATED pre-correction key(s) {sorted(legacy)} present in "
                "`sim` while the corrected environment is required. Remove them and set "
                f"{list(CORRECTED_REQUIRED_KEYS)} instead (env-correction design R8)."
            )
        missing = [key for key in CORRECTED_REQUIRED_KEYS if key not in sim]
        if missing:
            raise EnvConfigError(
                f"env config: corrected environment required but `sim` is missing "
                f"{sorted(missing)}. The adapter would silently fall back to the "
                "pre-correction primitive (legacy fixed-step move, hold=0), so the "
                "launch is refused instead."
            )
        for key in CORRECTED_REQUIRED_KEYS:
            if sim[key] is None:
                raise EnvConfigError(
                    f"env config: `sim.{key}` is null; the corrected environment "
                    "requires an explicit value."
                )

    params = _env_signature(env_cls).parameters
    kwargs: dict[str, Any] = {"n_envs": _coerce_int(n_envs)}
    for key in whitelist:
        if key in sim:
            try:
                kwargs[key] = _coercer(params[key].annotation)(sim[key])
            except EnvConfigError:
                raise
            except (TypeError, ValueError) as error:
                raise EnvConfigError(
                    f"env config: `sim.{key}` = {sim[key]!r} is not coercible to "
                    f"{params[key].annotation}"
                ) from error

    for key, value in (overrides or {}).items():
        if key not in whitelist and key not in _NON_SIM_PARAMS:
            raise EnvConfigError(
                f"env config: override {key!r} is not a {env_cls.__name__} parameter"
            )
        kwargs[key] = value
    return kwargs


def assert_corrected_env(env: Any, kwargs: dict[str, Any]) -> None:
    """Post-construction proof that the *built* adapter is the corrected one.

    The resolver guarantees the inputs; this guarantees the object.  Both
    are needed: a future caller that constructs ``DLOLabEnv`` directly would
    bypass :func:`resolve_env_kwargs` entirely, and the whole point of this
    change is that a pre-correction adapter must not be reachable silently
    from the training path.
    """
    if not getattr(env, "quasi_static", False):
        raise EnvConfigError(
            "env config: constructed adapter is NOT quasi-static (pre-correction "
            "primitive active). Refusing to train on the uncorrected environment."
        )
    expected_v = float(kwargs["move_v_max"])
    actual_v = float(getattr(env, "move_v_max"))
    if actual_v != expected_v:
        raise EnvConfigError(
            f"env config: adapter move_v_max {actual_v!r} != configured {expected_v!r}"
        )
    expected_hold = int(kwargs["move_hold_max_steps"])
    actual_hold = int(getattr(env, "move_hold_max_steps"))
    if actual_hold != expected_hold:
        raise EnvConfigError(
            f"env config: adapter move_hold_max_steps {actual_hold!r} != "
            f"configured {expected_hold!r}"
        )


def format_effective_env_params(env: Any, kwargs: dict[str, Any], *, env_cls: type) -> str:
    """Human- and grep-readable dump of the EFFECTIVE adapter parameters.

    Prints every constructor parameter with its resolved value and where the
    value came from (`config` when the `sim` block supplied it, `default`
    when the adapter signature did).  A launch that reads legacy physics is
    then visible in the first ten lines of the log instead of being
    indistinguishable from a corrected one.
    """
    params = _env_signature(env_cls).parameters
    lines = [
        "env-params: EFFECTIVE DLOLabEnv configuration "
        f"(quasi_static={bool(getattr(env, 'quasi_static', False))})"
    ]
    for name in ("n_envs", *allowed_sim_keys(env_cls)):
        if name in kwargs:
            source = "config"
            value = kwargs[name]
        else:
            source = "default"
            default = params[name].default
            value = None if default is inspect.Parameter.empty else default
        actual = getattr(env, name, "<not-exposed>")
        lines.append(f"env-params:   {name} = {value!r} (source={source}, adapter={actual!r})")
    return "\n".join(lines)
