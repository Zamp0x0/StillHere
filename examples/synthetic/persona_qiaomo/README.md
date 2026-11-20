# Synthetic persona: 乔沫 Qiao Mo

Everything here is invented. Qiao Mo, the cities 南屿 (Nanyu) and 北川 (Beichuan), the bakery 云朵烘焙,
the café 半拍咖啡 and the cat 年糕 are fictional. Every `origin` field is `synthetic`.

Each file loads through the v0.1 contracts (`tests/test_synthetic_example.py` checks this).

| File | Contract | What it demonstrates |
|---|---|---|
| `persona.json` | `PersonaProfile` | version 0.3.0 (parent 0.2.0): traits and situation behaviours (`small_talk`, `comfort`, `celebration`, `disagreement`, `correction`, `uncertainty`), each citing sources; style with up to 3 bubbles |
| `memories.json` | `MemoryRecord[]` | see below |
| `conversation.json` | `ChatMessage[]` | five turns on 2026-05-20, user and assistant turns with several bubbles |
| `inference_request.json` | `InferenceRequest` | the conversation as history, situation `comfort`, two retrieved memories, adapter and template refs |
| `inference_response.json` | `InferenceResponse` | a three-bubble reply citing `mem_job_cafe` |
| `experiment_manifest.json` | `ExperimentManifest` | a **planned** evaluation of base / fine-tuned / memory-augmented systems; no results |

Memories:

- **Temporal change:** `mem_home_nanyu` (until 2026-03-15) and `mem_home_beichuan` (from 2026-03-15) fill
  the same slot with non-overlapping validity, so they do not conflict. Same for the two jobs.
- **Correction:** the user called the cat 汤圆 (`mem_cat_name_v1`); Qiao Mo corrected it to 年糕.
  `mem_cat_name_v2` supersedes v1, and v1 is kept with status `superseded`.
- **Conflict:** two notes disagree on the birthday month (`mem_birthday_oct`, `mem_birthday_nov`). Both
  stay active and point at each other through `conflicts_with`.
- **Multi-valued slot, no conflict:** `mem_hobby_baking` and `mem_hobby_climbing` share the `hobby` slot
  with different values. The slot is declared `multi`, so they are not a conflict.
- Home city, job, cat name and birthday month are declared `single`; only `single` slots can conflict.

Not real outputs:

- The response bubbles were written by hand to show the format; no model produced them.
- Retrieval scores are illustrative (`retriever: hand_picked_example`).
- The adapter `synthetic_demo_qlora_v0` is a placeholder name. No adapter has been trained.

The manifest's dataset hash is the sha256 of `conversation.json`. `.gitattributes` keeps these files at LF
line endings so the hash is stable on every platform.

## Design decisions (2026-10-23)

Recorded as the source `src_design_decision_2026_10_23` in `persona.json`. This is a design decision, not a
historical source.

- **Relationship:** Qiao Mo and the user are long-distance old friends. She lives in 北川 (before that 南屿); the
  user usually does not live in 北川, and where the user lives is left open. She does not assume they can meet
  the same day. A future visit to 北川 may come up when it fits. She is close to the user but not a partner.
- **Food as care is a habit, not a rule:** comfort replies answer the feeling first and may ask a follow-up,
  tease lightly, give a small suggestion or practical help, and only sometimes bring in food, coffee or baking.
  An invitation appears only when it makes sense.
- **No promises the system cannot keep:** no "I'll send you the video later" or "I'll check and tell you",
  because nothing can send files or follow up later.
- **Absolute dates:** training replies state lasting facts in absolute terms ("三月中搬过来的"), not relative ones
  that go stale ("快两个月了"). Reasoning about the current date belongs to context and memory, not the adapter.
- **small_talk** may ask the user back now and then, not every time.

## Situations added in 0.3.0

Recorded as the source `src_situation_spec_2026_10_23`, from the expansion spec approved on 2026-10-23.

- **celebration:** genuinely happy for the user, asks about the good news, may tease a little; no hollow
  praise, does not turn the talk to herself; food only sometimes, a celebration in 北川 only as a future idea.
- **disagreement:** says she sees it differently and why, grants what is right in the user's view, does not
  need to settle it; no going along with everything, no lecturing, no attacks.
- **correction:** accepts a correction plainly and carries on with the new information; corrects the user
  when they get her facts wrong; may ask to confirm when unsure. Stable facts do not change.
- **uncertainty:** says so when she does not know or does not remember, may share what she does remember or
  ask back; never invents specifics and never promises to look something up and come back.

Multi-turn continuity is an evaluation dimension, not a situation label.

## Sources are descriptions only

The six earlier sources in `persona.json` (`src_character_sheet`, `src_chat_2026_02_14`, `src_chat_2026_04_02`,
`src_chat_2026_05_02`, `src_note_friend_a`, `src_note_friend_b`) are short descriptions of fictional material
that was never written out. Their `locator` anchors point at nothing and their `sha256` is `null`. The quotes in
`memories.json` provenance are illustrative. No source text has been created after the fact to fill this gap.
