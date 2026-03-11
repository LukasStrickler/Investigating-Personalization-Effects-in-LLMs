"""Helpers for building prompt specs and experiment grids."""

from __future__ import annotations

from itertools import product

from inference.experiments.types import ExperimentGrid, PromptSpec


def _validate_static_request(
    static_turns: list[dict[str, str]] | None,
    static_turns_list: list[list[dict[str, str]]] | None,
    final_user_message: str | None,
    final_user_messages: list[str] | None,
) -> None:
    """Ensure exactly one mode for static and one for request."""
    static_one = static_turns is not None
    static_multi = static_turns_list is not None
    if static_one and static_multi:
        raise ValueError(
            "Use either static_turns (one for all) or static_turns_list (multiple), not both."
        )
    req_one = final_user_message is not None
    req_multi = final_user_messages is not None
    if req_one and req_multi:
        raise ValueError(
            "Use either final_user_message (one) or final_user_messages (multiple), not both."
        )
    if not req_one and not req_multi:
        raise ValueError("Set either final_user_message or final_user_messages.")


def build_experiment_grid(
    *,
    system_prompt: str | None = None,
    system_prompts: list[str] | None = None,
    system_prompt_by_model: dict[str, str] | None = None,
    static_turns: list[dict[str, str]] | None = None,
    static_turns_list: list[list[dict[str, str]]] | None = None,
    final_user_message: str | None = None,
    final_user_messages: list[str] | None = None,
    model_aliases: list[str],
    run_cells: set[tuple[str, str]] | None = None,
) -> ExperimentGrid:
    """Build a standard experiment grid with one combined helper.

    Row = full prompt spec (JSON); cols = model_aliases; cells = response or not_requested.
    Exactly one mode per dimension:

    - **System:** one of system_prompt (one for all), system_prompts (multiple for all),
      system_prompt_by_model (per-model; system applied at run time, not in row spec).
    - **Static:** none (omit both), static_turns (one for all), or static_turns_list (multiple).
    - **Request:** final_user_message (one) or final_user_messages (multiple).

    Returns ExperimentGrid(prompts, model_aliases, run_cells). Pass run_cells to
    ExperimentConfig for sparse runs; omit to run all cells.
    """
    _validate_static_request(
        static_turns, static_turns_list, final_user_message, final_user_messages
    )

    # Resolve system dimension: per-model -> no system in spec; multiple -> list; one -> [system_prompt]
    if system_prompt_by_model is not None:
        systems: list[str | None] = []  # per-model: system not in row spec
    elif system_prompts is not None:
        systems = list(system_prompts)
    else:
        systems = [system_prompt]

    # Static dimension
    if static_turns_list is not None:
        statics: list[list[dict[str, str]] | None] = list(static_turns_list)
    elif static_turns is not None:
        statics = [static_turns]
    else:
        statics = [None]

    # Request dimension
    if final_user_messages is not None:
        requests = list(final_user_messages)
    elif final_user_message is not None:
        requests = [final_user_message]
    else:
        raise ValueError("final_user_message or final_user_messages is required")

    specs: list[PromptSpec] = []
    if not systems:
        # Per-model: no system in row spec
        for static, req in product(statics, requests):
            if req is None:
                raise ValueError("final_user_message / final_user_messages must not contain None")
            if static is None:
                specs.append(req)
            else:
                messages = list(static) + [{"role": "user", "content": req}]
                specs.append({"messages": messages})
    else:
        for sys, static, req in product(systems, statics, requests):
            if req is None:
                raise ValueError("final_user_message / final_user_messages must not contain None")
            if static is None:
                if sys is not None and (sys.strip() if isinstance(sys, str) else True):
                    specs.append({"system": sys, "user": req})
                else:
                    specs.append(req)
            else:
                messages = list(static) + [{"role": "user", "content": req}]
                spec: dict = {"messages": messages}
                if sys is not None and (sys.strip() if isinstance(sys, str) else True):
                    spec["system"] = sys
                specs.append(spec)

    return ExperimentGrid(
        prompts=specs,
        model_aliases=list(model_aliases),
        run_cells=run_cells,
    )
