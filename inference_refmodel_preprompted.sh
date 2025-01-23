#!/bin/bash -l


python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 1 --num_episodes_trained $2 --CD_tokens "action34" --r_illegal 6 --option $3 --ref_value_only "yes" --value "Ut";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 2 --num_episodes_trained $2 --CD_tokens "action34" --r_illegal 6 --option $3 --ref_value_only "yes" --value "Ut";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 3 --num_episodes_trained $2 --CD_tokens "action34" --r_illegal 6 --option $3 --ref_value_only "yes" --value "Ut";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 5 --num_episodes_trained $2 --CD_tokens "action34" --r_illegal 6 --option $3 --ref_value_only "yes" --value "Ut";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 6 --num_episodes_trained $2 --CD_tokens "action34" --r_illegal 6 --option $3 --ref_value_only "yes" --value "Ut";

python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 1 --num_episodes_trained $2 --CD_tokens "action34" --r_illegal 6 --option $3 --ref_value_only "yes" --value "De";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 2 --num_episodes_trained $2 --CD_tokens "action34" --r_illegal 6 --option $3 --ref_value_only "yes" --value "De";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 3 --num_episodes_trained $2 --CD_tokens "action34" --r_illegal 6 --option $3 --ref_value_only "yes" --value "De";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 5 --num_episodes_trained $2 --CD_tokens "action34" --r_illegal 6 --option $3 --ref_value_only "yes" --value "De";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 6 --num_episodes_trained $2 --CD_tokens "action34" --r_illegal 6 --option $3 --ref_value_only "yes" --value "De";
