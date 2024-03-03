import tensorflow as tf

tf.compat.v1.disable_eager_execution()


## ================ config ================


DIR = "/workspace/cpp/models"
MODEL_DIR = "/workspace/fullframe/result/540x540"

MODELS = ["e-unet_20220629_AutoLearning_fold5_576x576"]

## ========================================


def save_graph():
    """
    @機能：
    @引数：
    @戻値：
    """
    
    for model in MODELS:
        init_op = tf.compat.v1.initialize_all_variables()
        _ = tf.compat.v1.Variable(initial_value='fake_variable')
        sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, f"{MODEL_DIR}/{model}.ckpt")
        tf.train.write_graph(sess.graph.as_graph_def(), f"{MODEL_DIR}/", f"{model}.pb")
        
        del sess, saver


def main():
    save_graph()


if(__name__ == "__main__"):
    main()
