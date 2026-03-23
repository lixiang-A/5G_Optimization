# 5G Optimization Project

## Repository layout
- 背景信息/: contest statement, appendix, modeling markdown file, references, and earlier modeling materials
- BS2等5个文件/: data files for Question 3 and related readme notes
- channel_data等2个文件/: data files for Question 1, txt notes, and initial solving script
- channel_data等2个文件(1)/: data files for Question 2 and txt notes
- README.md: project overview

## Project priority
This project is for high-quality modeling, reinforcement-learning-based solving, and paper-oriented output.
Model quality comes first. Code and writing must serve the model.

## Core rules
1. Before answering any task, first read the relevant files in 背景信息/.
2. Do not invent parameters, assumptions, formulas, data sources, or innovations without support from repo files or explicit citations.
3. If information is missing, state the gap clearly instead of guessing.
4. When using data, always name the exact file being used.
5. Before modifying any file, state which file(s) will be changed and why.
6. Prefer small, reviewable changes. Do not touch unrelated files.
7. If the conversation becomes long, re-check the actual files before making claims.

## Modeling rules
1. Modeling rigor is the top priority. Never trade rigor for speed.
2. For each sub-question, check whether the model is logically closed, internally consistent, and aligned with the contest task.
3. Always distinguish clearly among given data, assumptions, derived quantities, and design choices.
4. Do not add assumptions casually. Every important assumption must be necessary and explainable.
5. Point out any missing links, weak justifications, unsupported innovations, or possible errors immediately.
6. Any innovation must remain understandable, defensible, and suitable for undergraduate research writing.
7. If a model has a hard flaw, identify and fix it before extending the analysis.

## Reinforcement learning rules
1. Any RL design must clearly define state, action, reward, transition logic, and training objective.
2. These parts must match the actual contest goals and data structure.
3. Do not create arbitrary reward functions just to make training run.
4. Explain why the RL formulation is reasonable, not only how it is coded.
5. Code should aim for correct, stable, convergent, and interpretable results.
6. If convergence or performance is poor, diagnose the cause instead of hiding the issue.
7. Do not present code as successful unless the result is supported by actual outputs or checks.

## Code rules
1. Main language: Python.
2. Run scripts from the project root directory whenever possible.
3. Use explicit relative paths for input files.
4. Do not rewrite working scripts unless necessary.
5. When editing a script, explain which script is changed, what is changed, why it is needed, and how to run it.
6. If a script depends on packages or specific file paths, say so clearly.
7. Save new outputs to new files whenever possible. Do not overwrite source files without a clear reason.

## Data rules
1. Treat original Excel and source data files as default read-only materials.
2. Do not modify original data files unless absolutely necessary.
3. If engineering adjustment is unavoidable, create a new derived file instead of overwriting the source file.
4. Any data adjustment must be minimal, explainable, traceable, and explicitly reported.
5. Never silently change data to force a better result.

## Writing and output rules
1. Output by sub-question whenever possible.
2. Keep answers accurate, structured, and logically strong.
3. Highlight important points clearly.
4. Use bullet points when they improve readability.
5. Brief analogies are allowed only when they improve understanding without reducing precision.
6. Tone should remain calm, practical, and rigorous.
7. Do not use vague praise or empty reassurance. Focus on what is correct, questionable, missing, or improvable.

## File safety
Files that can be modified:
- 5G环境建模相关 markdown files
- solving scripts such as .py files for each question

Files that must be modified cautiously:
- Excel data files

Files that must not be modified:
- contest statement document
- appendix document
- original reference materials unless explicitly requested

## Definition of done
A task is done only when:
1. the answer matches the actual repository files,
2. the modeling logic is checked for rigor and completeness,
3. any uncertainty or weakness is explicitly pointed out,
4. modified files are clearly identified,
5. and the result is easy to review, run, or integrate into later paper writing.
