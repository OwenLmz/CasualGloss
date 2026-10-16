# Third-party assets and release boundaries

Checked against primary project/model pages on 2026-10-16. This is an asset/access record, not legal advice or permission to relicense upstream work. The current upstream card and the paper's pinned revision can differ; keep both identities when obtaining assets.

| Asset | Primary source / recorded condition | Repository handling |
|---|---|---|
| How2Sign annotations/data | [Official site](https://how2sign.github.io/): CC BY-NC 4.0; attribution, noncommercial use, indicate modifications. The site separately says it cannot redistribute underlying How2 data. | No blanket MIT license on transcripts or derivatives. Use the official access route; record transformations and source CSV hash. Raw How2 audio is not bundled. |
| CLERC Epee | [Dataset card](https://huggingface.co/datasets/CLERC-DATA/epee): CC BY-NC-SA 4.0; paper uses revision `93bc8aa0a6526e86af80db5f59af8d133292c9b5`. | Keep attribution and the revision; transformed human-gloss manifests are separate data assets. Not relabeled as permissively licensed code. |
| Qwen2.5-7B-Instruct | [Model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct): Apache-2.0. Paper revision `a09a35458c702b33eeacc393d103063234e8bc28`. | Download upstream backbone separately. A LoRA adapter is not the full backbone; its training-data conditions still matter. |
| SeamlessStreaming | [Official repository](https://github.com/facebookresearch/seamless_communication): code has MIT components; SeamlessStreaming **model** is CC BY-NC 4.0. | Do not confuse code license with model license. External dependency, not vendored. |
| Qwen3-TTS / translation assets | Exact identities/configs are retained in `configs/speech/` and the pretrained-model registry. Output-redistribution conditions remain asset-specific. | Private author-data attachments do not create a new public license for synthetic speech or translated source material. |
| Python packages and external model sources | Individual upstream licenses | Referenced as dependencies; installed environments, vendor packages and caches are not redistributed. |

The authors have not yet selected a new blanket open-source license for their own code in this preparation. Existing notices are retained. Availability of code to read does not by itself confer an OSI license. Choose and approve the author's code license before advertising the project as open source; do not apply that license to data or upstream models.

Private SSH keys, tokens, server login configuration, unrelated projects, original How2Sign video/audio, submitted PDF, base-model weights, and environment directories are outside the Git source release. Final-paper derived data, selected LoRA, saved predictions and required synthetic audio are packaged as checksummed attachments to the same PRIVATE repository. Download requires the repository owner's authorized access and does not grant further redistribution rights. Official base models are obtained separately at the recorded revisions. The authorized server is retained as an unchanged original, not the only intended acquisition route.
