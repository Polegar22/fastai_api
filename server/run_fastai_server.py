import flask
from fastai.conv_learner import *
from fastai.dataset import *

app = flask.Flask(__name__)


def load_model():
    global arch, learn, sz
    arch = resnet34
    sz = 224
    data = ImageClassifierData.from_paths("data/uglybeauty/", bs=16, tfms=tfms_from_model(arch, sz))
    learn = ConvLearner.pretrained(arch, data, precompute=False)
    learn.load('beautyDetector_all')


def prepare_image(nparr):
    trn_tfms, val_tfms = tfms_from_model(arch, sz)
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    image = cv2.cvtColor(cv2.imdecode(nparr, flags).astype(np.float32) / 255, cv2.COLOR_BGR2RGB)
    image = val_tfms(image)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
            nparr = np.fromstring(flask.request.data, np.uint8)
            image = prepare_image(nparr)
            preds = learn.predict_array(image[None])

            if(np.argmax(preds) < 0.5):
                data["result"] = "This is beautiful"
            else:
                data["result"] = "This is ugly"

            print(np.argmax(preds))
            data["probability"] = int(np.argmax(preds))
            data["success"] = True

    return flask.jsonify(data)


if __name__ == "__main__":
    print(("Loading fastai..."))
    load_model()
    app.run()
