import { useEffect, useRef, useState } from "react";
import { useGlobalContext } from "../hooks/GlobalContext";
import NavBar from "../components/utility/NavBar";
import Footer from "../components/utility/Footer";
import gif from "../assets/waiting.gif";
import Sticker from "../components/utility/h4b";
import { analyzeFile, try_sample } from "../lib/api";
import samples from "../components/static/samples";
import open from "../assets/redirect.svg";
import background from "../assets/bg.png";
import Header from "../components/utility/Header";
import warning from "../assets/warning.png";
import Skeleton, { SkeletonTheme } from "react-loading-skeleton";
import "react-loading-skeleton/dist/skeleton.css";

const Playground = () => {
  const { isMenuOpen } = useGlobalContext();
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileType, setFileType] = useState("image");
  const [result, setResult] = useState(null);
  const hiddenFileInput = useRef(null);
  const [position, setPosition] = useState(null);
  const [type, setType] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const handleClick = (e) => {
    hiddenFileInput.current.click();
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    setUploadedFile(file);
    setFileName(file.name);
    if (file.type.startsWith("image/")) setFileType("image");
    else if (file.type.startsWith("video/")) setFileType("video");
    setResult(null);
    setType(null);
    setPosition(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true); // start loading
    console.log("Submit Started!");

    try {
      if (position && type) {
        console.log(position, type);
        const response = await try_sample(position, type);
        setResult(response);
        console.log(response);
      } else if (uploadedFile) {
        const response = await analyzeFile(uploadedFile);
        setResult(response);
        console.log(response);
      }
    } catch (err) {
      console.log("Error!");
    } finally {
      setSubmitting(false);
    }
  };

  const handleImage = async (index) => {
    setResult(null);
    setFileName(samples[index].name);
    const name = samples[index].name.split("_");
    console.log(name);
    let idx = parseInt(name[2]) - 1;
    console.log(idx);
    setPosition(idx);
    console.log(position);
    if (name.includes("fake") && name.includes("video")) {
      setType("video fake");
      setType("video");
    } else if (name.includes("real") && name.includes("image")) {
      setType("image real");
      setType("image");
    } else if (name.includes("fake") && name.includes("image")) {
      setType("image fake");
      setType("image");
    } else if (name.includes("real") && name.includes("video")) {
      setType("video real");
      setType("video");
    }
    console.log(fileType);
  };

  return (
    <>
      {/*
      <div className="relative h-full w-full bg-slate-950">
        <div className="absolute bottom-0 left-0 right-0 top-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]"></div>*/}
      <div className="h-[100vh] w-[100vw] bg-black">
        <NavBar />
        <Header />
        {isMenuOpen ? (
          <div className="h-screen w-screen z-20 fixed inset-0 backdrop-brightness-75"></div>
        ) : (
          <></>
        )}
        <Sticker />
        <div className="relative flex flex-col w-[100vw] h-[80vh] top-[10%] justify-center items-center">
          <div className="text-white flex flex-col justify-center items-center max-w-[80vw] text-center space-y-5 inter-400">
            <h1 className="text-3xl md:text-5xl">
              Detect Deepfakes using Vision Transformers and GAN Classifiers
            </h1>
            <h2 className="text-xl md:text-3xl bg-gradient-to-b from-slate-600 to-neutral-50 bg-clip-text text-transparent">
              [LOPT]
            </h2>
            <h3 className="text-sm md:text-md">
              Drag and drop your file or click to browse. Virtus will analyze it
              using our AI model and return a detailed report in seconds. Detect
              AI-generated content from models like Midjourney, DALL-E, Stable
              Diffusion, DeepFaceLab and FaceSwap.
            </h3>
          </div>
          <div className="min-w-[60vw] h-[30vh] md:min-h-[35vh] mt-4 border-[0.08rem] border-white rounded-xl overflow-hidden flex flex-col justify-between bg-black">
            <div className="w-full min-h-[1vw] bg-white"></div>
            <div className="min-h-[20vh] flex flex-row not-md:flex-col justify-center items-center">
              <div>
                <p className="text-sm text-white inter-400 text-center md:hidden">
                  Please Upload a file to detect deepfake:
                </p>
                <p className="text-md text-white inter-400 text-center not-md:hidden">
                  Try sample image:
                </p>
                <div className="grid grid-cols-3 text-sm text-white inter-400 space-x-3 space-y-2 not-md:hidden">
                  {samples.map((item, idx) => (
                    <ul key={idx}>
                      <div className="w-[10vw] border-[0.08rem] rounded-xl border-white p-1 flex flex-row justify-center gap-2">
                        <button
                          onClick={() =>
                            window.open(
                              item.source,
                              "_blank",
                              "noopener, noreferrer"
                            )
                          }
                        >
                          <img src={open} height={20} width={20}></img>
                        </button>
                        <button onClick={() => handleImage(idx)}>
                          {item.name}
                        </button>
                      </div>
                    </ul>
                  ))}
                </div>
              </div>
              <div className="h-20 flex justify-end items-end">
                <button
                  onClick={handleClick}
                  className="text-md text-white inter-400 border-[0.08rem] border-white rounded-2xl px-4 py-2 h-16"
                >
                  {!fileName ? <p>Upload File</p> : <p>{fileName}</p>}
                </button>
                <input
                  type="file"
                  accept="image/*,video/*"
                  ref={hiddenFileInput}
                  onChange={handleFileUpload}
                  style={{ display: "none" }}
                />
              </div>
            </div>
            <div className="min-h-[5vh] bg-white flex items-center justify-between">
              <div className="flex flex-row space-x-3">
                <img
                  src={gif}
                  height={14}
                  width={20}
                  className="ml-5 not-md:hidden"
                ></img>
                <img
                  src={gif}
                  height={12}
                  width={15}
                  className="ml-5 md:hidden"
                ></img>
                <p className="not-md:text-xs text-sm text-black inter-400">
                  WAITING FOR YOUR INPUT
                </p>
              </div>
              <div className="w-20 md:w-32 h-8 mr-1 md:mr-5 bg-white rounded-2xl border-[0.08rem] border-black flex flex-row text-center justify-between items-center">
                {fileType === "image" ? (
                  <>
                    <div className=" bg-black w-[50%] h-full rounded-2xl text-center text-white text-xs flex justify-center items-center border-[0.08rem] border-black">
                      Image
                    </div>
                    <div className="text-xs mr-4">Video</div>
                  </>
                ) : (
                  <>
                    <div className="text-xs ml-4 flex justify-center items-center">
                      Image
                    </div>
                    <div className=" bg-black w-[50%] h-full rounded-2xl text-center text-white text-xs flex justify-center items-center border-[0.08rem] border-black">
                      Video
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
          <div className="mt-5 flex flex-row justify-center items-center space-x-5">
            {result ? (
              <>
                <div className="w-[15vw] h-[1vh] bg-white rounded-md overflow-hidden not-md:hidden">
                  <div
                    className={
                      result.label === "fake"
                        ? `h-[1vh] bg-[#f03b05] rounded-m`
                        : `h-[1vh] bg-green-500 rounded-m`
                    }
                    style={{ width: `${result.confidence}%` }}
                  />
                </div>
                <span className="text-md inter-400 text-white">
                  CONFIDENCE :
                  <span
                    className={
                      result.label === "fake"
                        ? "text-[#f03b05]"
                        : "text-green-500"
                    }
                  >
                    {" "}
                    {result.confidence}%
                  </span>
                </span>
                <span className="text-md inter-400 text-white">
                  LIKELY{" "}
                  <span
                    className={
                      result.label === "fake"
                        ? "text-[#f03b05]"
                        : "text-green-500"
                    }
                  >
                    {result.label.toUpperCase()}
                  </span>
                </span>
              </>
            ) : (
              <></>
            )}
            {result || (position && type) ? null : submitting ? (
              <div className="text-white text-md inter-400 flex items-center space-x-2">
                <p>Analyzing your {fileType}</p>
              </div>
            ) : (
              <button
                className="text-md text-white inter-400 bg-[#f03b05] px-4 py-2 rounded-2xl"
                onClick={handleSubmit}
              >
                Check for DeepFake!
              </button>
            )}
          </div>
          <div className="relative w-screen flex justify-center items-center mt-3 gap-2">
            <img src={warning} height={24} width={24}></img>
            <p className="text-sm text-white">
              We are currently facing issues in our samples route. Please upload
              an image from your local machine for seamless experience.
            </p>
          </div>
        </div>
        <Footer />
      </div>
      {/*</></div>*/}
    </>
  );
};

export default Playground;
