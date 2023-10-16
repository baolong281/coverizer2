import "./App.css";
import { useDropzone } from "react-dropzone";
import { useCallback } from "react";
import { useState } from "react";
import { ReactSlider } from "react-slider";

function App() {
  const [uploaded, setUploaded] = useState(false);
  const [image, setImage] = useState(null);
  const [outputRes, setOutputRes] = useState(512);
  const [resolutionError, setResolutionError] = useState("");

  const onDrop = useCallback((acceptedFiles) => {
    setUploaded(true);
    setImage(acceptedFiles[0]);
  }, []);

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  const handleSubmit = () => {

     const body = {
        image: image,
         text: "hello"
      };

      console.log(fetch("inference"));
      let res = fetch("/inference", {
        method: "POST",
          headers: {
            "Content-Type": "application/json",
            },
        body: JSON.stringify(body),
      }
      ).then((res) => console.log(res.body));
  };

  const handleResolutionChange = (e) => {
    if(Number.isInteger(+e.target.value) && e.target.value > 0 && e.target.value < 2560){
      setOutputRes(e.target.value);
      setResolutionError(false)
      return;
    }
    else if (e.target.value > 2560) {
        setResolutionError("please enter a resolution less than 2560");
        return;
    }
    setResolutionError("please enter a valid resolution");
  }

  return (
    <div className="App">
      <div className="title">coverizer 2</div>
      <div className="images">
        <div>
          <img src={process.env.PUBLIC_URL + "manalive.png"} alt="expanded" />
        </div>
        <div>
          <img
            src={process.env.PUBLIC_URL + "lovelessexp.png"}
            alt="expanded"
          />
        </div>
      </div>
      <div className="inference">
        <div {...getRootProps()} className="drop">
          <input {...getInputProps()} />
          {!uploaded ? (
            <div className="dropbox">
              drop or select image
              <button>upload</button>
            </div>
          ) : (
            <>
              <img
                src={URL.createObjectURL(image)}
                alt="uploaded"
                className="image"
              />
              <button>resubmit</button>
            </>
          )}
        </div>
        <div className="right">
          <p>output resolution</p>
          <input onChange={handleResolutionChange}></input>
            {resolutionError ? <p className="error">{resolutionError}</p> : null}
          <div>
            <button onClick={handleSubmit} className="submit">submit</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
