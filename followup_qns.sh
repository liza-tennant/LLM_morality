#!/bin/bash -l

python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail _PT2 --run_idx 1 --num_episodes_trained 1000 --CD_tokens "action34" --r_illegal 6 --option "COREDe" --follow_up_qn_IPD "yes";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail _PT3 --run_idx 1 --num_episodes_trained 1000 --CD_tokens "action34" --r_illegal 6 --option "COREDe" --follow_up_qn_IPD "yes";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail _PT3 --run_idx 1 --num_episodes_trained 1000 --CD_tokens "action34" --r_illegal 6 --option "COREUt" --follow_up_qn_IPD "yes";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail _PT4 --run_idx 1 --num_episodes_trained 1000 --CD_tokens "action34" --r_illegal 6 --option "COREDe" --follow_up_qn_IPD "yes";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail _PT3after2 --run_idx 1 --num_episodes_trained 500 --CD_tokens "action34" --r_illegal 6 --option "COREDe" --follow_up_qn_IPD "yes";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail _PT3after2 --run_idx 1 --num_episodes_trained 500 --CD_tokens "action34" --r_illegal 6 --option "COREUt" --follow_up_qn_IPD "yes";
