#!/bin/bash

START_ITERATION=1 # the starting iteration
NUM_ITERATIONS=1000
INPUT_FOLDER="github/2.1.ExtractingKeywords/TEMP_Iterations"
SCRIPTS_FOLDER="github/2.1.ExtractingKeywords"
LOG_FILE="github/iteration_log.txt"

echo "Resuming iteration run from iteration $START_ITERATION at $(date)" >> "$LOG_FILE"

# Loop through each iteration starting from START_ITERATION
for (( i=START_ITERATION; i<=START_ITERATION+NUM_ITERATIONS-1; i++ ))
do
    echo "Processing iteration $i..." | tee -a "$LOG_FILE"
    INPUT_ITERATION=$i
    OUTPUT_ITERATION=$((i+1))

    mkdir -p "${INPUT_FOLDER}/${OUTPUT_ITERATION}"

    # Run the LRR_batching script for the current iteration
    python3 "${SCRIPTS_FOLDER}/LRR_TNT_KID.py" "${INPUT_FOLDER}/${i}/${i}.filtered_llm_papers.json" "${INPUT_FOLDER}/${i}/${i}.filtered_non_llm_papers.json" 2>&1 | tee -a "$LOG_FILE"

    if [ "$i" -eq $((START_ITERATION + NUM_ITERATIONS - 1)) ]; then
        echo "Completed all iterations." | tee -a "$LOG_FILE"
        break
    fi

    # Run the Iterative2SetsCreation script for the current iteration
    python3 "${SCRIPTS_FOLDER}/2.Iterative2SetsCreation_automatic.py" \
        --input-folder "${INPUT_FOLDER}/${i}" \
        --input-iteration "$i" \
        --output-folder "${INPUT_FOLDER}/${OUTPUT_ITERATION}" \
        --output-iteration "$OUTPUT_ITERATION" 2>&1 | tee -a "$LOG_FILE"
done

echo "Iteration process completed at $(date)" | tee -a "$LOG_FILE"
