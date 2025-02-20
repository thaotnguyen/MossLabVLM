python eval.py
python csv_annotator.py BLIP2_results.csv
python csv_annotator.py GPT-4V_results.csv
python csv_annotator.py LLaVa_results.csv
python csv_annotator.py DeepSeek_VL2_results.csv
cd LLaVA-Med
python -m llava.eval.model_vqa
cd ..
python csv_annotator.py LLaVA-Med/llava-med_results.csv
