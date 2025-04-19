import { useEffect, useState } from "react";
import { useGlobalContext } from "../hooks/GlobalContext";
import NavBar from "../components/utility/NavBar";
import Footer from "../components/utility/Footer";
import gif from "../assets/waiting.gif";
import Sticker from "../components/utility/h4b";
import analyzeFile from "../lib/api";

const Playground = () => {
  const { isMenuOpen } = useGlobalContext();
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileType, setFileType] = useState("image");
  const [result, setResult] = useState(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    setUploadedFile(file);
    if (file.type.startsWith("image/")) setFileType("image");
    else if (file.type.startsWith("video/")) setFileType("video");
    setResult(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!uploadedFile) return;
    try {
      const response = await analyzeFile(uploadedFile);
      setResult(response);
      console.log(response);
    } catch (err) {
      console.log("Error!");
    }
  };

  return (
    <>
      <div className="h-[100vh] w-[100vw] bg-black">
        <NavBar />
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
          <div className="min-w-[60vw] min-h-[35vh] mt-4 border-[0.08rem] border-white rounded-xl overflow-hidden flex flex-col justify-between">
            <div className="w-full min-h-[1vw] bg-white"></div>
            <div className="min-h-[20vh] flex flex-col justify-center items-center">
              <p className="text-sm text-white inter-400">Try sample image:</p>
              <input
                type="file"
                accept="image/*,video/*"
                onChange={handleFileUpload}
                className="max-h-[10vh] not-md:max-w-[40vw] not-md:text-xs px-4 py-2 border border-gray-300 rounded-md text-white"
              />
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
              <div className="w-32 h-8 mr-5 bg-white rounded-2xl border-[0.08rem] border-black flex flex-row text-center justify-between items-center">
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
                <div className="w-[15vw] h-[1vh] bg-white rounded-md overflow-hidden">
                  <div
                    className={
                      result.label === "fake"
                        ? `w-[${Math.floor(
                            result.confidence
                          )}%] h-[1vh] bg-[#f03b05] rounded-m`
                        : `w-[${Math.floor(
                            result.confidence
                          )}%] h-[1vh] bg-green-500 rounded-m`
                    }
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
            {result ? (
              <></>
            ) : (
              <>
                <button
                  className="text-md text-white inter-400 bg-[#f03b05] px-4 py-2 rounded-2xl"
                  onClick={handleSubmit}
                >
                  Check for DeepFake!
                </button>
              </>
            )}
          </div>
        </div>
        <Footer />
      </div>
    </>
  );
};

export default Playground;
