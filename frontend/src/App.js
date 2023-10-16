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
  const [outputImage, setOutputImage] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    setUploaded(true);
    setImage(acceptedFiles[0]);
  }, []);
  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  const encodeImageFileAsURL = (file) => {
    let reader = new FileReader();
    reader.onloadend = () => {};
    reader.readAsDataURL(file);
    return reader.result;
  };

  const handleSubmit = () => {
    if (!image) {
      alert("please upload an image");
      return;
    }

    const reader = new FileReader();

    reader.onloadend = (e) => {
      const image_binary = e.target.result;
      const body = {
        image: image_binary,
        res: outputRes,
      };

      let res = fetch("api/inference", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      }).then((res) => {
        res.json().then((data) => {
            setOutputImage(data.filename);
            console.log(data.filename);
        });
      });
    };
    reader.readAsDataURL(image);
  };

  const handleResolutionChange = (e) => {
    if (
      Number.isInteger(+e.target.value) &&
      e.target.value > 0 &&
      e.target.value < 2560
    ) {
      setOutputRes(e.target.value);
      setResolutionError(false);
      return;
    } else if (e.target.value > 2560) {
      setResolutionError("please enter a resolution less than 2560");
      return;
    }
    setResolutionError("please enter a valid resolution");
  };

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
        <div>
          <img
            src={process.env.PUBLIC_URL + "nurture.png"}
            alt="expanded"
          />
        </div>
        <div>
          <img
            src={process.env.PUBLIC_URL + "alvvays.png"}
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
            <button onClick={handleSubmit} className="submit">
              generate
            </button>
          </div>
        </div>
      </div>
      <div className="output">
        {outputImage ? (
            <>
            <p>ai output</p>
            <img src={outputImage} alt="output" />
            </>
        ) : (<></>)}
      </div>
    </div>
  );
}

export default App;
