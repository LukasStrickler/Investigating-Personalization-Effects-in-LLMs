real_conversation_histories
============================
Pipeline that extracts real WildChat conversations with explicit gender
evidence and turns them into personas for the project.

Dataset reference:
    Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., & Deng, Y. (2024).
    WildChat: 1M ChatGPT Interaction Logs in the Wild. ICLR 2024.
    https://huggingface.co/datasets/allenai/WildChat-1M

FINAL FILE TO USE: wildchat_personas.jsonl
(one JSON record per kept conversation, full turns, gender-labeled persona)

WORKFLOW
--------
1. Run conversation_histories_extraction.py
   -> produces wildchat_conversations.jsonl + wildchat_gender_evidence_results.csv
      (and a copy, wildchat_gender_evidence_results_checked.csv)

2. Open wildchat_gender_evidence_results_checked.csv by hand and fill the
   "correct" column: 1 = keep this row's gender label, 0 = drop it.

3. Run tojson.py
   -> reads the checked CSV, joins each kept id's full conversation from
      wildchat_conversations.jsonl, and writes wildchat_personas.jsonl /
      wildchat_personas.csv.

SCRIPTS
-------
conversation_histories_extraction.py
    Streams the WildChat-1M dataset, filters to English / non-toxic /
    reasonable-length first prompts, and scores each conversation's first
    user message for explicit gender evidence (self-identification,
    shorthand like "25F", gendered roles, contextual requests). Outputs:
      - wildchat_conversations.jsonl : full conversation turns, keyed by
        conversation_id (needed by tojson.py so it doesn't have to
        re-stream WildChat).
      - wildchat_gender_evidence_results.csv /
        wildchat_gender_evidence_results_checked.csv : one row per
        conversation with gender evidence, with an empty "correct" column
        for manual review.

tojson.py
    Step 3 of the workflow. Reads the hand-checked CSV, keeps only rows
    marked correct==1, and builds one persona record per kept conversation
    (joining full turns from wildchat_conversations.jsonl). Outputs:
      - wildchat_personas.jsonl : the final dataset, one JSON object per
        conversation ({history_id, persona, combination_ids, messages,
        generated_at}).
      - wildchat_personas.csv : same conversations, one row each, with a
        flattened transcript column (for quick spreadsheet review).

FILES
-----
wildchat_conversations.jsonl           full WildChat conversations (intermediate, large)
wildchat_gender_evidence_results.csv          gender-evidence rows, unreviewed
wildchat_gender_evidence_results_checked.csv  same, hand-marked with "correct" column
wildchat_personas.jsonl                FINAL: personas dataset used downstream
wildchat_personas.csv                  FINAL (spreadsheet view of the same data)
