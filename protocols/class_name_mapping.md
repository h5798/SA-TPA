# CLIP class-name mapping

The mapping is frozen at feature-extraction time and is identical for ViT-B/32 and ViT-B/16.

## Rule

Directory names are converted by replacing underscores with spaces, except for the three explicit Office-31 aliases below. Letter case is preserved by the tokenizer but is not used to create dataset-specific synonyms.

| Dataset label | CLIP text |
|---|---|
| `back_pack` | `backpack` |
| `bike` | `bicycle` |
| `phone` | `telephone` |

Examples covered by the general underscore rule include `Alarm_Clock` -> `Alarm Clock`, `Desk_Lamp` -> `Desk Lamp`, `File_Cabinet` -> `File Cabinet`, `Paper_Clip` -> `Paper Clip`, `Postit_Notes` -> `Postit Notes`, and `Trash_Can` -> `Trash Can`.

No class-name remapping may be changed during Adaptive SA-TPA development. Any later semantic-name study must regenerate features under a distinct run tag and be reported as a separate prompt ablation.
