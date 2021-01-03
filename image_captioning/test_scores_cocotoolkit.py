import json
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap

MODEL = "SAR_bert"
MULTILEVEL_ATTENTION =True
if __name__ == "__main__":

    test_path = "dataset_splits/TEST_COCOTOOLKIT_FORMAT.json"
    if MODEL == "BASELINE":
        generated_sentences_path = "baselinebeam5.json"
        scores_path = "baseline_coco_resultsbeam5.json"
    elif MODEL == "SAR_avg":
        generated_sentences_path = "SAR_avgbeam5.json"
        scores_path = "SAR_avg_coco_resultsbeam5.json"
    elif MODEL == "SAR_norm":
        if MULTILEVEL_ATTENTION:
            generated_sentences_path = "SAR_normbeam5.json"
            scores_path = "SAR_norm_coco_resultsbeam5.json"
        else:
            generated_sentences_path = "SAR_norm_no_multiattentionbeam5.json"
            scores_path = "SAR_norm_no_multiattention_coco_resultsbeam5.json"
    elif MODEL == "SAR_bert":
        generated_sentences_path = "SAR_bertbeam5.json"
        scores_path = "SAR_bert_coco_resultsbeam5.json"
    elif MODEL == "SAR_norm_wt_m":
        generated_sentences_path = "SAR_norm_wt_mbeam5.json"
        scores_path = "SAR_norm_wt_m_coco_resultsbeam5.json"
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
