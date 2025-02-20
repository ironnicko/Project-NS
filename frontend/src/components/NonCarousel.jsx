import Carousel from "react-bootstrap/Carousel";
import React, { useState, useEffect } from "react";
import kicks from "./kicks.json";
import { initializeApp } from "firebase/app";
import {
  getFirestore,
  collection,
  where,
  getDocs,
  query,
  QuerySnapshot,
  orderBy,
} from "firebase/firestore";
// import firebaseConfig from "./serviceAccountKey.json";
import TimePicker from "react-time-picker";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_apiKey,
  authDomain: import.meta.env.VITE_authDomain,
  projectId: import.meta.env.VITE_projectId,
  storageBucket: import.meta.env.VITE_storageBucket,
  messagingSenderId: import.meta.env.VITE_messagingSenderId,
  appId: import.meta.env.VITE_appId,
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

console.log(import.meta.env);

const epochTimeConverted = (time, hm) => {
  const date = new Date(time + " " + hm);
  const unixTimestamp = Math.floor(date.getTime() / 1000);
  return unixTimestamp;
};

export default function NonCarousel() {
  const dConstant = new Date();
  const [index, setIndex] = useState(0);
  const [loader, setLoader] = useState(false);
  const [imageData, setImageData] = useState([kicks]);
  const [date, setDate] = useState({
    to: `${dConstant.getFullYear()}-${
      dConstant.getMonth() + 1
    }-${dConstant.getDate()}`,
    from: "1970-1-01",
  });
  const [timeF, onChangeF] = useState("00:00");
  const [timeT, onChangeT] = useState("00:00");

  const handleSelect = (selectedIndex, e) => {
    setIndex(selectedIndex);
  };

  const floodData = async () => {
    const imagesRef = collection(db, "Images");
    const q = query(
      imagesRef,
      where("time", ">=", epochTimeConverted(date.from, timeF)),
      where("time", "<=", epochTimeConverted(date.to, timeT)),
      orderBy("time", "desc")
    );
    const querySnapshot = await getDocs(q);
    const temp = [];
    querySnapshot.forEach((doc) => {
      temp.push(doc.data());
    });
    setLoader(false);
    setImageData(temp);
  };

  useEffect(() => {
    if (loader) floodData();
    return () => console.clear();
  });

  const handleSubmit = (event) => {
    event.preventDefault();
    setLoader(true);
  };
  return (
    <div>
      <form
        className="date-form d-flex justify-content-center"
        method="post"
        action="#"
        onSubmit={handleSubmit}
      >
        <div className="from col-md-4 d-flex fs-4">
          <label for="from">From</label>
          <input
            type="date"
            className="form-control"
            id="from"
            name="From"
            onChange={(e) => {
              setDate({ from: e.target.value, to: date.to });
            }}
          />
        </div>
        <TimePicker value={timeF} onChange={onChangeF} />

        <div className="col-md-4 to d-flex fs-4">
          <label for="to">To</label>
          <input
            type="date"
            className="form-control"
            id="to"
            name="to"
            onChange={(e) => {
              setDate({ from: date.from, to: e.target.value });
            }}
          />
        </div>
        <TimePicker value={timeT} onChange={onChangeT} />

        <div>
          <input type="submit" className="btn btn-dark submit-btn-form"></input>
        </div>
      </form>
      <Carousel
        activeIndex={index}
        onSelect={handleSelect}
        className="cara text-dark"
        interval={null}
      >
        {loader && <h1 className="text-center">Loading...</h1>}
        {imageData.map((queryItem) => (
          <Carousel.Item>
            <div className="row text-center">
              <div className="col-lg-6 text-center p-0">
                <img
                  className="d-block cara-img"
                  src={`data:image/png;base64,${
                    queryItem["original_image"] || queryItem["segmented_image"]
                  }`}
                  alt="First slide"
                />
                <h3>Original</h3>
              </div>
              <div className="col-lg-6 p-0 text-center">
                <img
                  className="d-block cara-img"
                  src={`data:image/png;base64,${
                    queryItem["segmented_image"] || queryItem["originial_image"]
                  }`}
                  alt="First slide"
                />
                <h3>Segmented</h3>
              </div>
              <div className="w-100 text-center nc-div">
                <h3 className="nuclei-counts bg-dark text-light rounded w-50">
                  Nuclei Count : {queryItem["adjusted_nuclei_count"]} approx.
                </h3>
              </div>
              <h4>
                Slide : {index + 1} / {imageData.length}
              </h4>
            </div>
          </Carousel.Item>
        ))}
      </Carousel>
      {!imageData.length && (
        <h1 className="text-center">Nothing to see here 👀</h1>
      )}
    </div>
  );
}
