import { useState, useEffect } from "react"
import data from "./fakeNews"
import "./App.scss"
function App() {
  const [textSize, setTextSize] = useState("");
  const [oznitelikCıkarımı, setOznitelikCikarimi] = useState("");
  const [algoritmaName, setAlgoritmaName] = useState("");
  const [destekV, setDestekV] = useState("");
  const [ysaGk, setYsaGk] = useState("");
  const [ysaAf, setYsaAf] = useState("");
  const [knnK, setKnnK] = useState("");
  const [knnM, setKnnM] = useState("");

  const [dvmBlock, setDvmBlock] = useState(false);
  const [ysaBlock, setYsaBlock] = useState(false);
  const [knnBlock, setKnnBlock] = useState(false);

  const [result, setResult] = useState(null);


  const handleCheckboxChangeSize = (event) => {
    const { id } = event.target;
    setTextSize(id);
  };

  const handleCheckboxChangeOznitelik = (event) => {
    const { id } = event.target;
    setOznitelikCikarimi(id);
  };

  const handleCheckboxChangeDestekV = (event) => {
    const { id } = event.target;
    setDestekV(id);
  };

  const handleCheckboxChangeAf = (event) => {
    const { id } = event.target;
    setYsaAf(id);
  };

  const handleCheckboxChangeGk = (event) => {
    const { id } = event.target;
    setYsaGk(id);
  };
  const handleCheckboxChangeK = (event) => {
    const { id } = event.target;
    setKnnK(id);
  };
  const handleCheckboxChangeM = (event) => {
    const { id } = event.target;
    setKnnM(id);
  };


  const openBox = () => {
    if (algoritmaName === "Destek Vektör Makineleri") {
      setYsaGk("")
      setYsaAf("")
      setKnnK("")
      setKnnM("")
      setDvmBlock(true)
    } else {
      setDestekV("")
      setYsaGk("")
      setYsaAf("")
      setKnnK("")
      setKnnM("")
      setDvmBlock(false)
    }
    if (algoritmaName === "Yapay Sinir Ağları") {
      setYsaGk("")
      setYsaAf("")
      setYsaBlock(true)

    } else {
      setDestekV("")
      setYsaGk("")
      setYsaAf("")
      setKnnK("")
      setKnnM("")
      setYsaBlock(false)
    }
    if (algoritmaName === "K-En Yakın Komşu") {
      setDestekV("")
      setYsaGk("")
      setYsaAf("")
      setKnnBlock(true)
    } else {
      setDestekV("")
      setYsaGk("")
      setYsaAf("")
      setKnnK("")
      setKnnM("")
      setKnnBlock(false)
    }
  }

  const handleCheckboxChangeAlgoritma = (event) => {
    const { id } = event.target;
    setAlgoritmaName(id);
    openBox()
  };


  useEffect(() => {
    openBox();
  }, [algoritmaName, result]);


  const findData = () => {
    const result = data.find((item) => {
      // searchItem ile item arasındaki her özelliği karşılaştırıyoruz
      return (
        item.textSize === textSize &&
        item.oznitelikCıkarımı === oznitelikCıkarımı &&
        item.algoritmaName === algoritmaName &&
        item.destekV === destekV &&
        item.ysaGk === ysaGk &&
        item.ysaAf === ysaAf &&
        item.knnK === knnK &&
        item.knnM === knnM
      );
    });
    console.log(textSize)
    console.log(oznitelikCıkarımı)
    console.log(algoritmaName)
    console.log(destekV)
    console.log(ysaGk)
    console.log(ysaAf)
    console.log(knnK)
    console.log(knnM)
    console.log(result)
    setResult(result);
  }

  return (
    <div className="container">
      <div className="main-header">Sahte Haber Tespiti</div>
      <div className="block-1">
        <div className="custom-checkbox-size">
          <p className="header">Test Size</p>
          <div className="checkbox-item">
            <input
              type="checkbox"
              id="0.2"
              name="0.2"
              checked={textSize === '0.2'}
              onChange={handleCheckboxChangeSize}
            />
            <label htmlFor="item1">0.2</label>
          </div>
          <div className="checkbox-item">
            <input
              type="checkbox"
              id="0.3"
              name="0.3"
              checked={textSize === '0.3'}
              onChange={handleCheckboxChangeSize}
            />
            <label htmlFor="item2">0.3</label>
          </div>
        </div>
        <div className="custom-checkbox-nitelik">
          <p className="header">Öznitelik Çıkarımı</p>
          <div className="checkbox-item">
            <input
              type="checkbox"
              id="bagOfWord"
              name="bagOfWord"
              checked={oznitelikCıkarımı === 'bagOfWord'} // Sadece seçili checkbox işaretli olur
              onChange={handleCheckboxChangeOznitelik}
            />
            <label htmlFor="item1">Bag Of Words</label>
          </div>
          <div className="checkbox-item">
            <input
              type="checkbox"
              id="tf_ıdf"
              name="tf_ıdf"
              checked={oznitelikCıkarımı === 'tf_ıdf'}
              onChange={handleCheckboxChangeOznitelik}
            />
            <label htmlFor="item2">TF-IDF</label>
          </div>
        </div>
        <div className="custom-checkbox-algoritma">
          <p className="header">Makine Öğrenmesi Algoritması</p>
          <div className="checkbox-item">
            <input
              type="checkbox"
              id="Lojistik Regresyon"
              name="Lojistik Regresyon"
              checked={algoritmaName === 'Lojistik Regresyon'}
              onChange={handleCheckboxChangeAlgoritma}
            />
            <label htmlFor="item2">Lojistik Regresyon</label>
          </div>
          <div className="checkbox-item">
            <input
              type="checkbox"
              id="Karar Ağaçları"
              name="Karar Ağaçları"
              checked={algoritmaName === 'Karar Ağaçları'}
              onChange={handleCheckboxChangeAlgoritma}
            />
            <label htmlFor="item2">Karar Ağaçları</label>
          </div>
          <div className="checkbox-item">
            <input
              type="checkbox"
              id="Destek Vektör Makineleri"
              name="Destek Vektör Makineleri"
              checked={algoritmaName === 'Destek Vektör Makineleri'}
              onChange={handleCheckboxChangeAlgoritma}
            />
            <label htmlFor="item2">Destek Vektör Makineleri</label>
          </div>
          <div className="checkbox-item">
            <input
              type="checkbox"
              id="Yapay Sinir Ağları"
              name="Yapay Sinir Ağları"
              checked={algoritmaName === 'Yapay Sinir Ağları'}
              onChange={handleCheckboxChangeAlgoritma}
            />
            <label htmlFor="item2">Yapay Sinir Ağları</label>
          </div>
          <div className="checkbox-item">
            <input
              type="checkbox"
              id="K-En Yakın Komşu"
              name="K-En Yakın Komşu"
              checked={algoritmaName === 'K-En Yakın Komşu'}
              onChange={handleCheckboxChangeAlgoritma}
            />
            <label htmlFor="item2">K-En Yakın Komşu</label>
          </div>
        </div>
        <div>
          <button onClick={() => findData()}>Göster</button>
        </div>
      </div>
      <div className="block1-block2">
        <div className="block-2">
          <div className="dvm-block">
            {dvmBlock
              ? (
                <div className="custom-checkbox-dvm">
                  <p className="header">Destek Vektör Makineleri için</p>
                  <div className="checkbox-item">
                    <input
                      type="checkbox"
                      id="linear"
                      name="linear"
                      checked={destekV === 'linear'} // Sadece seçili checkbox işaretli olur
                      onChange={handleCheckboxChangeDestekV}
                    />
                    <label htmlFor="item1">linear</label>
                  </div>
                  <div className="checkbox-item">
                    <input
                      type="checkbox"
                      id="rbf"
                      name="rbf"
                      checked={destekV === 'rbf'}
                      onChange={handleCheckboxChangeDestekV}
                    />
                    <label htmlFor="item2">rbf</label>
                  </div>
                </div>
              )
              : (<div></div>)}
          </div>
          <div className="ysa-block">
            {ysaBlock
              ? (
                <div className="custom-checkbox-ysa">
                  <p className="header">Yapay Sinir Ağları</p>
                  <div className="ysa-gk">
                    <p className="header-1">Gizli Katman Sayısı</p>

                    <div className="checkbox-item">
                      <input
                        type="checkbox"
                        id="100"
                        name="100"
                        checked={ysaGk === '100'} // Sadece seçili checkbox işaretli olur
                        onChange={handleCheckboxChangeGk}
                      />
                      <label htmlFor="item1">100</label>
                    </div>
                    <div className="checkbox-item">
                      <input
                        type="checkbox"
                        id="200"
                        name="200"
                        checked={ysaGk === '200'}
                        onChange={handleCheckboxChangeGk}
                      />
                      <label htmlFor="item2">200</label>
                    </div>
                  </div>
                  <div className="ysa-af">
                    <p className="header-1">Aktivasyon Fonksiyonu</p>

                    <div className="checkbox-item">
                      <input
                        type="checkbox"
                        id="tanh"
                        name="tanh"
                        checked={ysaAf === 'tanh'} // Sadece seçili checkbox işaretli olur
                        onChange={handleCheckboxChangeAf}
                      />
                      <label htmlFor="item1">tanh</label>
                    </div>
                    <div className="checkbox-item">
                      <input
                        type="checkbox"
                        id="logistic"
                        name="logistic"
                        checked={ysaAf === 'logistic'}
                        onChange={handleCheckboxChangeAf}
                      />
                      <label htmlFor="item2">logistic</label>
                    </div>
                  </div>
                </div>
              )
              : (<div></div>)}
          </div>
          <div className="knn-block">
            {knnBlock
              ? (
                <div className="custom-checkbox-knn">
                  <p className="header">K-en Yakın Komşu</p>
                  <div className="ysa-gk">
                    <p className="header-1">K</p>
                    <div className="checkbox-item">
                      <input
                        type="checkbox"
                        id="3"
                        name="3"
                        checked={knnK === '3'} // Sadece seçili checkbox işaretli olur
                        onChange={handleCheckboxChangeK}
                      />
                      <label htmlFor="item1">3</label>
                    </div>
                    <div className="checkbox-item">
                      <input
                        type="checkbox"
                        id="5"
                        name="5"
                        checked={knnK === '5'}
                        onChange={handleCheckboxChangeK}
                      />
                      <label htmlFor="item2">5</label>
                    </div>
                    <div className="checkbox-item">
                      <input
                        type="checkbox"
                        id="13"
                        name="13"
                        checked={knnK === '13'}
                        onChange={handleCheckboxChangeK}
                      />
                      <label htmlFor="item2">13</label>
                    </div>
                  </div>
                  <div className="ysa-af">
                    <p className="header-1">Metric</p>
                    <div className="checkbox-item">
                      <input
                        type="checkbox"
                        id="cosine"
                        name="cosine"
                        checked={knnM === 'cosine'} // Sadece seçili checkbox işaretli olur
                        onChange={handleCheckboxChangeM}
                      />
                      <label htmlFor="item1">cosine</label>
                    </div>
                    <div className="checkbox-item">
                      <input
                        type="checkbox"
                        id="euclidean"
                        name="euclidean"
                        checked={knnM === 'euclidean'}
                        onChange={handleCheckboxChangeM}
                      />
                      <label htmlFor="item2">euclidean</label>
                    </div>
                    <div className="checkbox-item">
                      <input
                        type="checkbox"
                        id="manhattan"
                        name="manhattan"
                        checked={knnM === 'manhattan'}
                        onChange={handleCheckboxChangeM}
                      />
                      <label htmlFor="item2">manhattan</label>
                    </div>
                  </div>
                </div>
              )
              : (<div className="not"></div>)}
          </div>
        </div>
        <div className={result ? `block-3-result` : `not`}>
          {result
            ?
            (<div className="block-3">
              <img src={result.imgPath} alt="Sonuç Resmi" />
              <div>
                <div>
                  <p>Accuracy: {result.outPut.accuracy}</p>
                  <p>Precision: {result.outPut.precision}</p>
                </div>
                <div>
                  <p>Recall: {result.outPut.recall}</p>
                  <p>Score :{result.outPut.score}</p>
                </div>
              </div>
            </div>)
            : (<div className="not">Sonuç Yok</div>)
          }
        </div>
      </div>
    </div >
  )
}

export default App
