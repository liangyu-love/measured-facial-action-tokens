import json, os, sys
print(json.load(open(os.path.expanduser("~/work/msa-pilot/outputs/full/expert_tuning.json")))["text_best"][sys.argv[1]])
