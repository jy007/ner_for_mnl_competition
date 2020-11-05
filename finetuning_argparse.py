from argparse import ArgumentParser

def get_argparse():
    parse = ArgumentParser()

    # 定于需要的命令行参数
    parse.add_argument("--task_name",default=None,type=str,required=True,
                        help="the name of the task to train select in the list")
    parse.add_argument("--data_dir",default=None,type=str,required=True,
                        help="the input data dir, should contain the training files for CoNLL-2003 NER  task")
    parse.add_argument("--model_type",default=None,type=str,required=True,
                        help="model type select in the list: [bert,albert]")
    parse.add_argument("--model_name_or_path",default=None,type=str,required=True,
                        help="path to pre-trained model or shortcut name select in the list")
    parse.add_argument("--output_dir",default=None,type=str,required=True,
                        help="the output directory where the model predictions and checkpoints will be written")
    #Other parameters
    parse.add_argument("--markup",default="bios",type=str,choices=["bios","bio"])
    parse.add_argument("--loss_type",default="ce",type=str,choices=["lsr","focal","ce"])
    parse.add_argument("--config_name",default="",type=str,help="pretrained config name or path if not he same as model_name")
    parse.add_argument("--tokenizer_name",default="",type=str,help="pretrained tokenizer name or path if not same as model_name")
    parse.add_argument("--cache_dir",default="",type=str,help="where do you want to store the pre-trained models downloaded from s3")
    parse.add_argument("--train_max_seq_length",default=128,type=int,help="the maximum total input sequence length ")
    parse.add_argument("--eval_max_seq_length",default=512,type=int,help="the maximum total input sequence length after tokenization") #这里与训练不一致会有什么问题吗
    parse.add_argument("--do_train",action="store_true",help="whether to run training")
    parse.add_argument("--do_eval",action="store_true",help="whether to run eval on the dev set.")
    parse.add_argument("--do_predict",action="store_true",help="whether to run predictions on the test set")
    parse.add_argument("--evaluate_during_training",action="store_true",help="whether to run evaluation during training at each logging step")
    parse.add_argument("--do_lower_case",action="store_true",help="set this flag if you are using an uncased model")

    # adversarial training
    parse.add_argument("--do_adv",action="store_true",help="whether to adversarial training")
    parse.add_argument("--adv_epsilon",default=1.0,type=float,help="Epsilon for adversarial")
    parse.add_argument("--adv_name",default="word_embeddings",type=str,help="name for adversarial layer")
    
    parse.add_argument("--per_gpu_train_batch_size",default=8,type=int,help="Batch size per GPU/CPU for training")
    parse.add_argument("--per_gpu_eval_batch_size",default=8,type=int,help="Batch size per GPU/CPU for evaluation")
    parse.add_argument("--gradient_accumulation_steps",type=int,default=1,help="number of updates steps to accumulate before preforming a backward/update pass.")
    parse.add_argument("--learning_rate",default=5e-5,type=float,help="het initial learning rate for Adam")
    parse.add_argument("--crf_learning_rate",default=5e-5,type=float,help="the initial learning rate for crf and  linear layer")
    parse.add_argument("--weight_decay",default=0.01,type=float,help="weight decay if we apply some")
    parse.add_argument("--adam_epsilon",default=1e-8,type=float,help="epsilon for adam optimizer")
    parse.add_argument("--max_grad_norm",default=1.0,type=float,help="max gradient norm.")
    parse.add_argument("--num_train_epochs",default=3.0,type=float,help="total number of training epochs to perform")
    parse.add_argument("--max_steps",default=-1,type=int,help="if >0: set total number of training steps to perform. Override num_train_epochs")
    
    parse.add_argument("--warmup_proportion",default=0.1,type=float,help="Proportation of training to perform linear learning rate warmup for E.g., 0.1 = 10% of training")
    parse.add_argument("--logging_steps",type=int,default=50,help="Log every X updates steps")
    parse.add_argument("--save_steps",type=int,default=50,help="save checkpoint every X updates steps")
    parse.add_argument("--eval_all_checkpoints",action="store_true",help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parse.add_argument("--predict_checkpoints",type=int,default=0,help="predict checkpoint starting with the same prefix as model_name ending and ending with step number")
    parse.add_argument("--no_cuda",action="store_true",help="avoid using CUDA when available")
    parse.add_argument("--overwrite_output_dir",action="store_true",help="overwrite the content of the output directory")
    parse.add_argument("--overwrite_cache",action="store_true",help="overwrite the cached training and evaluation sets")
    parse.add_argument("--seed",type=int,default=42,help="random seed for initialization")
    parse.add_argument("--fp16",action="store_true",help="whether to use 16-bit (mix) precision (through Nvidia apex) instend of 32-bit")
    parse.add_argument("--fp16_opt_leval",type=str,default="01",help="for fp16: apex amp optimization level selected in [00,01,02,03].,ses detaild at https://nvidia.githun.io/apex/amp.html")
    parse.add_argument("--local_rank",type=int,default=-1,help="for distributed training :local_rank")
    parse.add_argument("--server_ip",type=str,default="",help="for distant debugging")
    parse.add_argument("--server_port",type=str,default="",help="for distant debugging.")

    return parse