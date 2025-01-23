#!/bin/bash -l

#Note: the --opp_strat argument here refers to the training opponent, not the test-time opponent. The test-time opponent is hard-coded to Random in the inference script. 

#analysis with the new action tokens action3 and action4, order of presentation in the payoff matrix identical to the training data
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 1 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 ;
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 2 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 ;
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 3 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 ;
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 5 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 ;
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 6 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 ;


#analysis with the new action tokens action3 and action4, order of presentation in the payoff matrix permuted vs the training data
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 1 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "permuted1";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 2 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "permuted1";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 3 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "permuted1";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 5 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "permuted1";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 6 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "permuted1";

python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 1 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "permuted2";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 2 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "permuted2";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 3 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "permuted2";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 5 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "permuted2";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 6 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "permuted2";

python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 1 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "reversed";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 2 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "reversed";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 3 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "reversed";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 5 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "reversed";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 6 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "reversed";
python /src/inference_vsRandom.py --base_model_id "google/gemma-2-2b-it" --opp_strat "TFT" --PARTs_detail $1 --run_idx 1 --num_episodes_trained $2 --CD_tokens $3 --r_illegal 6 --option $4 --order_CD "reversed";

