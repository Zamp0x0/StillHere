# Persona

A **persona** in StillHere is a profile (one person in one chat export) plus a **persona card**: a structured description of how
that person writes. The card is rendered into the persona prompt (the "Persona Skill") that the chat model follows.

## What the automatic card contains

Built only from the target person's replies in the **train** split:

- **reply length:** quartiles in characters, how often they split a reply into several bubbles;
- **habits:** punctuation, laughter, emoji, question style, typical sentence endings;
- **distinctive expressions:** words this person uses clearly more often than the other person (log-odds);
- **frequent short replies:** for example "好" or "嗯嗯";
- **real example exchanges:** shown to the model as style examples, explicitly marked as not current facts;
- **AI identity rule:** the persona says it is an AI when sincerely asked; it is not the real person.

```bash
stillhere persona review --profile <profile-id>                  # summary + editable fields
stillhere persona review --profile <profile-id> --show-prompt    # the full prompt (local terminal only)
stillhere persona card show --profile <profile-id>               # prompt preview of the active card
```

## Editing

Only these fields can be edited:

- `display_name`;
- `ai_identity`;
- `user_confirmed_facts` (facts you state yourself, used as user-confirmed persona settings);
- `addressing_confirmed`;
- `extra_style_notes`;
- `avoid`.

The statistics are kept as computed.

```bash
stillhere persona review --profile <profile-id> --edit                     # $EDITOR / Notepad; saved as a new version
stillhere persona review --profile <profile-id> --from-file my_card.yaml   # an edited copy of cards/<card-id>/persona_card.yaml
stillhere persona review --profile <profile-id> --approve                  # mark the current card USER_REVIEWED, no changes
stillhere persona finalize --profile <profile-id>                          # chat uses the active card after finalize
```

Every card passes the same schema validation, whatever its source.

| Source | How it is made | `review_status` |
|---|---|---|
| AUTO | `quickstart` / `prepare` | `AUTO_GENERATED` |
| MANUAL | `review --edit` / `--from-file` | `USER_REVIEWED` |
| LLM_ASSISTED | `card assist --allow-external-api` | draft until you review it |

`card assist` sends the card's style statistics to the configured `card_assist` provider. It sends example replies, which are
private chat lines, only with `--send-examples`. Add `--activate` to make the draft the active card.

## Versions and rollback

```bash
stillhere persona card list --profile <profile-id>
stillhere persona card activate --profile <profile-id> --card <card-id>
stillhere persona finalize --profile <profile-id>
```

- Versions are never overwritten.
- Chat uses exactly one active card.
- After `activate`, run `finalize`; until then chat stops with `CONFIG_STALE` rather than using an unexpected card.

## Style without training

The default is **Persona Skill only**: the card's prompt on a general chat model, with the adapter off. A trained LoRA adapter is
optional (see [training.md](training.md)). In the project's own evaluation, the adapter did not reliably beat the prompt-only
persona, so it is never enabled automatically.
