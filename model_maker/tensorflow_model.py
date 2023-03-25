import tensorflow as tf
import pandas as pd
import numpy as np
import os

with open("data_more.csv", "r") as file:
    isWake = []
    claps = []
    for line in file:
        asList = line.split(",")
        isWake.append(1 if asList[-1].strip() == "true" else 0)
        claps.append(list(map(lambda numStr: int(numStr), asList[:-1])))

    # shuffle
    combined = list(zip(claps, isWake))
    np.random.shuffle(combined)
    claps, isWake = zip(*combined)

    MAX_LEN_CLAPS = 10

    features = pd.DataFrame(claps).to_numpy()
    np.nan_to_num(features, copy=False, nan=-1)

    labels = np.array(isWake)

    # padding to max_length
    padded_features = np.zeros((len(features), MAX_LEN_CLAPS))
    padded_features[:] = -1
    padded_features[:, :features.shape[1]] = features

    # make labels numerical
    labels = labels.astype(int)

    # split into train and test
    num_train = int(len(features) * 0.8)
    train_features = padded_features[:num_train]
    train_labels = labels[:num_train]
    test_features = padded_features[num_train:]
    test_labels = labels[num_train:]

    # create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(MAX_LEN_CLAPS,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_features, train_labels, epochs=50)

    # evaluate
    test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc}")


    # convert to tflite format
    BATCH_SIZE = 1
    STEPS = MAX_LEN_CLAPS
    INPUT_SIZE = 10

    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype)
    )

    os.system("rm -rf model")
    os.system("rm -f model_tflite")

    model.save("model", save_format="tf", signatures=concrete_func)

    converter = tf.lite.TFLiteConverter.from_saved_model("model")
    model_tflite = converter.convert()

    open("model_tflite", "wb").write(model_tflite)

    os.system("xxd -i model_tflite > model_tflite_micro.h")


