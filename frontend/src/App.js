import "./App.css";
import { useDropzone } from "react-dropzone";
import { useCallback } from "react";

function App() {


    const onDrop = useCallback((acceptedFiles) => {
        console.log(acceptedFiles);
    }, []);

    const { getRootProps, getInputProps } = useDropzone({onDrop});

  return (
    <div className="App">
      <div className="title">coverizer 2</div>
      <div className="images">
          <div>
            <img src={process.env.PUBLIC_URL + 'manalive.png'} alt="expanded" />
          </div>
          <div>
            <img src={process.env.PUBLIC_URL + 'lovelessexp.png'} alt="expanded" />
          </div>
      </div>
      <div {...getRootProps()} className="drop">
          <input {...getInputProps()} />
           drop or select image
        <button>upload</button>
      </div>
    </div>
  );
}

export default App;
