import json
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap

MODEL = "BASELINE"
if __name__ == "__main__":

    test_path = "dataset_splitsTEST_COCOTOOLKIT_FORMAT.json"
    if MODEL == "BASELINE":
        generated_sentences_path = "baseline.json"
        scores_path = "baseline_coco_results.json"
    elif MODEL == "NEAREST":
        generated_sentences_path = "nearest.json"
        scores_path = "nearest_coco_results.json"
    elif MODEL == "ATTENTION_NEAREST":
        generated_sentences_path = "SATN.json"
        scores_path = "SATN_coco_results.json"
    else:
        raise Exception("unknown model")

    coco = COCO(test_path)
    cocoRes = coco.loadRes(generated_sentences_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params["image_id"] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # save each image score and the avg score to a dict
    predicted = {"model": MODEL}
    individual_scores = [eva for eva in cocoEval.evalImgs]
    for i in range(len(individual_scores)):
        predicted[individual_scores[i]["image_id"]] = individual_scores[i]
    predicted["avg_metrics"] = cocoEval.eval

    # save scores dict to a json
    with open(scores_path, 'w+') as f:
        json.dump(predicted, f, indent=2)
