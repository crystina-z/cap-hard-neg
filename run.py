import os
from capreolus import constants, parse_config_string
from capreolus.utils.loginit import get_logger

# TODO: import all modules defined under capreolus_extensions/, e.g. 
# from capreolus_extensions.bertreranker import *

from utils import load_yaml
from args import get_args


logger = get_logger(__name__)


def run_single_fold(args, config):
    fold = config['fold']
    # task = WandbRerankerTask(config)
    task = RerankTask(config)

    if args.train:
        logger.info(f"TASK: {args.config_path}")
        logger.info(f"TRAINING ON FOLD {fold}")
        # preds, scores = task.train(init_path=args.init_path)
        preds, scores = task.train()
        scores = scores["fold_dev_metrics"]

    if args.eval:
        logger.info(f"TASK: {args.config_path}\tEVALUATING ON FOLD {fold}")
        task.predict_on_dev()
        scores = task.evaluate_on_dev()

    logger.info(f"dev metrics on fold {fold}: ")
    logger.info(scores)


def main():
    args = get_args()
    config = load_yaml(args.config_path)
    pretrain_dir = args.pretrained_dir
    if pretrain_dir != "":
        config["reranker"]["pretrained"] = os.path.join(pretrain_dir, config["reranker"]["pretrained"])
        config["reranker"]["extractor"]["tokenizer"]["pretrained"] = \
            os.path.join(pretrain_dir, config["reranker"]["extractor"]["tokenizer"]["pretrained"])

    record_commit_id(args.config_path)
    run_single_fold(args, config)


if __name__ == "__main__":
    main()