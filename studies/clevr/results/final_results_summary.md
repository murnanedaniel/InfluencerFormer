# CLEVR Final Results Summary

## Headline checks

- Best schedule run: `warmup_cosine` at `0.5495`.
- Best tau-only run: `τ=0.15` at `0.5625`.
- Best slot-only PM3 run: `PM3 (slotted)` at `0.6094`.
- Combined recipe winner: `Hungarian (slotted + warmup_cosine)` at `0.4314`.
- No-early-stop PM3 rerun: `0.6607`.
- Best 2x2x2 PM3 grid point: `PM3 tau=0.12 | slotted=False | warmup_cosine=True` at `0.5828`.

## Interpretation notes

- PM3 clearly benefits from either `warmup_cosine` alone or `τ=0.15` alone, but the full `τ=0.15 + slots + warmup` recipe is not additive.
- The PM3 grid points to `τ=0.12 + warmup_cosine + standard decoder` as the strongest clean PM3 recipe in this study (`0.5828`).
- Slot embeddings help PM3 when introduced on their own, but they appear to interact poorly with the stronger schedule / temperature recipe.
- Early stopping was a real confounder for the PM3 best-recipe run: removing it improved best distance from `0.7039` to `0.6607`.
- Even after fixing early stopping, the slotted+warmup+τ=0.15 PM3 recipe still underperforms the best single-factor PM3 variants, so the regression is not just truncation.
- Hungarian improves dramatically under the combined recipe (`0.4314`), which suggests the architectural/schedule changes are not universally bad; the issue is PM3-specific interaction, not a broken training stack.
- The PM3 grid and the no-early-stop rerun show some instability across nominally similar settings, so the combined recipe likely has a narrower optimization basin than the simpler PM3 variants.

## Cost comparison

- `Chamfer`: best `1.0893`, time to best `1.81 min`, time to `0.80` `not reached`, GPU `5.80 ms/batch`, wall `5.86 ms/batch`.
- `PM3 τ=0.12`: best `0.7010`, time to best `9.35 min`, time to `0.80` `2.82 min`, GPU `6.03 ms/batch`, wall `6.08 ms/batch`.
- `Hungarian`: best `0.7564`, time to best `25.61 min`, time to `0.80` `5.93 min`, GPU `138.53 ms/batch`, wall `138.59 ms/batch`.

## Suggested next steps

- Use the current CLEVR study as parameter discovery and mechanism analysis, not as the final community benchmark claim.
- Carry forward two PM3 candidates: the strongest clean PM3 point from the grid and the best single-factor schedule/tau wins.
- Use COCO DETR-style detection as the next benchmark for matching/cost relevance to the wider literature.