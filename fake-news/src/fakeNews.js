const data = [
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Lojistik Regresyon",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/bagOfWord/1-LR.png",
        "outPut": { "accuracy": "0.90", "precision": "0.90", "recall": "0.90", "score": "0.90" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Karar Ağaçları",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/bagOfWord/2-KA.png",
        "outPut": { "accuracy": "0.90", "precision": "0.90", "recall": "0.90", "score": "0.90" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Destek Vektör Makineleri",
        "destekV": "rbf",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/bagOfWord/3-DVM-rbf.png",
        "outPut": { "accuracy": "0.91", "precision": "0.92", "recall": "0.92", "score": "0.91" }
    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Destek Vektör Makineleri",
        "destekV": "linear",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/bagOfWord/3-DVM-linear.png",
        "outPut": { "accuracy": "0.93", "precision": "0.94", "recall": "0.94", "score": "0.94" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "100",
        "ysaAf": "logistic",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/bagOfWord/4-YSA-100-Logistic.png",
        "outPut": { "accuracy": "0.94", "precision": "0.95", "recall": "0.95", "score": "0.95" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "100",
        "ysaAf": "tanh",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/bagOfWord/4-YSA-100-Tanh.png",
        "outPut": { "accuracy": "0.94", "precision": "0.95", "recall": "0.95", "score": "0.95" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "200",
        "ysaAf": "logistic",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/bagOfWord/4-YSA-200-Logistic.png",
        "outPut": { "accuracy": "0.94", "precision": "0.94", "recall": "0.94", "score": "0.94" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "200",
        "ysaAf": "tanh",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/bagOfWord/4-YSA-100-Tanh.png",
        "outPut": { "accuracy": "0.94", "precision": "0.95", "recall": "0.95", "score": "0.95" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/bagOfWord/5-KNN-3-cosine.png",
        "knnK": "3",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.85", "precision": "0.86", "recall": "0.85", "score": "0.85" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/bagOfWord/5-KNN-3-euclidean.png",
        "knnK": "3",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.74", "precision": "0.76", "recall": "0.74", "score": "0.73" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/bagOfWord/5-KNN-3-manhattan.png",
        "knnK": "3",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.65", "precision": "0.76", "recall": "0.65", "score": "0.61" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/bagOfWord/5-KNN-5-cosine.png",
        "knnK": "5",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.85", "precision": "0.87", "recall": "0.86", "score": "0.85" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/bagOfWord/5-KNN-5-euclidean.png",
        "knnK": "5",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.73", "precision": "0.75", "recall": "0.73", "score": "0.72" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/bagOfWord/5-KNN-5-manhattan.png",
        "knnK": "5",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.63", "precision": "0.75", "recall": "0.63", "score": "0.58" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/bagOfWord/5-KNN-13-cosine.png",
        "knnK": "13",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.84", "precision": "0.86", "recall": "0.85", "score": "0.85" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/bagOfWord/5-KNN-13-euclidean.png",
        "knnK": "13",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.70", "precision": "0.75", "recall": "0.70", "score": "0.68" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/bagOfWord/5-KNN-13-manhattan.png",
        "knnK": "13",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.59", "precision": "0.75", "recall": "0.59", "score": "0.50" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Lojistik Regresyon",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/1-LR.png",
        "outPut": { "accuracy": "0.94", "precision": "0.95", "recall": "0.95", "score": "0.95" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Karar Ağaçları",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/2-KA.png",
        "outPut": { "accuracy": "0.89", "precision": "0.89", "recall": "0.89", "score": "0.72" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Destek Vektör Makineleri",
        "destekV": "rbf",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/3-DVM-linear.png",
        "outPut": { "accuracy": "0.51", "precision": "0.27", "recall": "0.52", "score": "0.35" }
    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Destek Vektör Makineleri",
        "destekV": "linear",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/3-DVM-rbf.png",
        "outPut": { "accuracy": "0.95", "precision": "0.95", "recall": "0.95", "score": "0.95" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "100",
        "ysaAf": "logistic",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/4-YSA-100-logistic.png",
        "outPut": { "accuracy": "0.96", "precision": "0.95", "recall": "0.95", "score": "0.95" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "100",
        "ysaAf": "tanh",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/4-YSA-100-tanh.png",
        "outPut": { "accuracy": "0.92", "precision": "0.94", "recall": "0.94", "score": "0.94" }
    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "200",
        "ysaAf": "logistic",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/4-YSA-200-logistic.png",
        "outPut": { "accuracy": "0.94", "precision": "0.95", "recall": "0.95", "score": "0.95" }
    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "200",
        "ysaAf": "tanh",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/4-YSA-1200-tanh.png",
        "outPut": { "accuracy": "0.94", "precision": "0.94", "recall": "0.94", "score": "0.94" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/5-KNN-3-cosine.png",
        "knnK": "3",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.85", "precision": "0.86", "recall": "0.85", "score": "0.85" }
    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/5-KNN-3-euclidean.png",
        "knnK": "3",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.53", "precision": "0.75", "recall": "0.53", "score": "0.38" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/5-KNN-3-manhattan.png",
        "knnK": "3",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.51", "precision": "0.27", "recall": "0.52", "score": "0.35" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/5-KNN-5-cosine.png",
        "knnK": "5",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.85", "precision": "0.86", "recall": "0.86", "score": "0.86" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/5-KNN-5-euclidean.png",
        "knnK": "5",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.52", "precision": "0.75", "recall": "0.53", "score": "0.37" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/5-KNN-5-manhattan.png",
        "knnK": "5",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.51", "precision": "0.27", "recall": "0.52", "score": "0.35" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/5-KNN-13-cosine.png",
        "knnK": "13",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.85", "precision": "0.86", "recall": "0.85", "score": "0.85" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/5-KNN-13-euclidean.png",
        "knnK": "13",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.52", "precision": "0.75", "recall": "0.52", "score": "0.36" }

    },
    {
        "textSize": "0.2",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/02/tf_ıdf/5-KNN-13-manhattan.png",
        "knnK": "13",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.51", "precision": "0.27", "recall": "0.52", "score": "0.35" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Lojistik Regresyon",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/bagOfWord/1-LR.png",
        "outPut": { "accuracy": "0.94", "precision": "0.95", "recall": "0.95", "score": "0.95" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Karar Ağaçları",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/bagOfWord/2-KA.png",
        "outPut": { "accuracy": "0.89", "precision": "0.90", "recall": "0.90", "score": "0.90" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Destek Vektör Makineleri",
        "destekV": "rbf",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/bagOfWord/3-DVM-linear.png",
        "outPut": { "accuracy": "0.91", "precision": "0.91", "recall": "0.91", "score": "0.91" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Destek Vektör Makineleri",
        "destekV": "linear",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/bagOfWord/3-DVM-rbf.png",
        "outPut": { "accuracy": "0.93", "precision": "0.93", "recall": "0.93", "score": "0.93" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "100",
        "ysaAf": "logistic",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/bagOfWord/4-YSA-100-logistic.png",
        "outPut": { "accuracy": "0.94", "precision": "0.94", "recall": "0.94", "score": "0.94" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "100",
        "ysaAf": "tanh",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/bagOfWord/4-YSA-100-tanh.png",
        "outPut": { "accuracy": "0.94", "precision": "0.94", "recall": "0.94", "score": "0.94" }
    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "200",
        "ysaAf": "logistic",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/bagOfWord/4-YSA-200-logistic.png",
        "outPut": { "accuracy": "0.94", "precision": "0.94", "recall": "0.94", "score": "0.94" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "200",
        "ysaAf": "tanh",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/bagOfWord/4-YSA-200-tanh.png",
        "outPut": { "accuracy": "0.94", "precision": "0.95", "recall": "0.95", "score": "0.95" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/bagOfWord/5-KNN-3-cosine.png",
        "knnK": "3",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.84", "precision": "0.86", "recall": "0.85", "score": "0.85" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/bagOfWord/5-KNN-3-euclidean.png",
        "knnK": "3",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.73", "precision": "0.75", "recall": "0.73", "score": "0.72" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/bagOfWord/5-KNN-3-manhattan.png",
        "knnK": "3",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.64", "precision": "0.75", "recall": "0.64", "score": "0.59" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/bagOfWord/5-KNN-5-cosine.png",
        "knnK": "5",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.84", "precision": "0.86", "recall": "0.85", "score": "0.84" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/bagOfWord/5-KNN-5-euclidean.png",
        "knnK": "5",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.72", "precision": "0.75", "recall": "0.72", "score": "0.71" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/bagOfWord/5-KNN-5-manhattan.png",
        "knnK": "5",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.62", "precision": "0.74", "recall": "0.62", "score": "0.56" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/bagOfWord/5-KNN-13-cosine.png",
        "knnK": "13",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.84", "precision": "0.86", "recall": "0.85", "score": "0.85" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/bagOfWord/5-KNN-13-euclidean.png",
        "knnK": "13",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.69", "precision": "0.75", "recall": "0.70", "score": "0.68" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "bagOfWord",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/bagOfWord/5-KNN-13-manhattan.png",
        "knnK": "13",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.57", "precision": "0.74", "recall": "0.58", "score": "0.48" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Lojistik Regresyon",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/1-LR.png",
        "outPut": { "accuracy": "0.94", "precision": "0.94", "recall": "0.94", "score": "0.94" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Karar Ağaçları",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/2-KA.png",
        "outPut": { "accuracy": "0.89", "precision": "0.89", "recall": "0.89", "score": "0.89" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Destek Vektör Makineleri",
        "destekV": "rbf",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/3-DVM-linear.png",
        "outPut": { "accuracy": "0.51", "precision": "0.26", "recall": "0.51", "score": "0.34" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Destek Vektör Makineleri",
        "destekV": "linear",
        "ysaGk": "",
        "ysaAf": "",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/3-DVM-rbf.png",
        "outPut": { "accuracy": "0.95", "precision": "0.95", "recall": "0.95", "score": "0.95" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "100",
        "ysaAf": "logistic",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/4-YSA-100-logistic.png",
        "outPut": { "accuracy": "0.94", "precision": "0.94", "recall": "0.94", "score": "0.94" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "100",
        "ysaAf": "tanh",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/4-YSA-100-tanh.png",
        "outPut": { "accuracy": "0.94", "precision": "0.94", "recall": "0.94", "score": "0.94" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "200",
        "ysaAf": "logistic",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/4-YSA-200-logistic.png",
        "outPut": { "accuracy": "0.94", "precision": "0.94", "recall": "0.94", "score": "0.94" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "Yapay Sinir Ağları",
        "destekV": "",
        "ysaGk": "200",
        "ysaAf": "tanh",
        "knnK": "",
        "knnM": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/4-YSA-200-tanh.png",
        "outPut": { "accuracy": "0.94", "precision": "0.94", "recall": "0.94", "score": "0.94" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/5-KNN-3-cosine.png",
        "knnK": "3",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.84", "precision": "0.85", "recall": "0.85", "score": "0.85" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/5-KNN-3-euclidean.png",
        "knnK": "3",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.52", "precision": "0.75", "recall": "0.52", "score": "0.37" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/5-KNN-3-manhattan.png",
        "knnK": "3",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.51", "precision": "0.75", "recall": "0.51", "score": "0.35" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/5-KNN-5-cosine.png",
        "knnK": "5",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.85", "precision": "0.86", "recall": "0.85", "score": "0.85" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/5-KNN-5-euclidean.png",
        "knnK": "5",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.51", "precision": "0.75", "recall": "0.36", "score": "0.95" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/5-KNN-5-manhattan.png",
        "knnK": "5",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.51", "precision": "0.75", "recall": "0.51", "score": "0.35" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/5-KNN-13-cosine.png",
        "knnK": "13",
        "knnM": "cosine",
        "outPut": { "accuracy": "0.85", "precision": "0.86", "recall": "0.85", "score": "0.85" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/5-KNN-13-euclidean.png",
        "knnK": "13",
        "knnM": "euclidean",
        "outPut": { "accuracy": "0.51", "precision": "0.75", "recall": "0.51", "score": "0.35" }

    },
    {
        "textSize": "0.3",
        "oznitelikCıkarımı": "tf_ıdf",
        "algoritmaName": "K-En Yakın Komşu",
        "destekV": "",
        "ysaGk": "",
        "ysaAf": "",
        "imgPath": "./src/assets/images/03/tf_ıdf/5-KNN-13-manhattan.png",
        "knnK": "13",
        "knnM": "manhattan",
        "outPut": { "accuracy": "0.51", "precision": "0.75", "recall": "0.51", "score": "0.35" }
    },
]

export default data