import { useGlobalContext } from "../hooks/GlobalContext";
import NavBar from "../components/NavBar";
import Footer from "../components/Footer";
import Sticker from "../components/h4b";
import background from "../assets/bg.png";
import dot from "../assets/dot3.svg";
import copy from "../assets/copy.svg";
import arrow from "../assets/arrow2.svg";
import github from "../assets/github.svg";
import { InfiniteCarousel } from "../components/Carousel";
import { useLocation } from "react-router";

function Home() {
  const { isMenuOpen, changeMenu } = useGlobalContext();

  return (
    <>
      <NavBar />
      <img className="absolute h-screen w-screen z-0" src={background}></img>
      {isMenuOpen ? (
        <div className="h-screen w-screen z-20 fixed inset-0 backdrop-brightness-75"></div>
      ) : (
        <></>
      )}
      <Sticker />
      <div className="fixed z-20 md:left-5 md:top-[70%] right-5 bottom-[15%] flex flex-col md:w-[2vw]">
        <button className="bg-[#1E1E1E] p-1">
          <img src={github} height={10} width={20}></img>
        </button>
      </div>
      <div className="fixed -right-0 top-[15%] w-[100%] md:w-[50vw] justify-between flex flex-row border-l-[0.09rem] border-[#1e1e1e]">
        <div>
          <InfiniteCarousel />
          <div className="not-md:ml-2 md:w-[50vw]">
            <div className="md:text-xl text-md inter-400 flex flex-row w-[90vw] md:w-[50vw] justify-between">
              <div className="flex flex-row">
                <img src={dot}></img>
                <span className="text-white">Truth,</span>
                <span className="text-red-500">Verified</span>
              </div>
              <div className="flex flex-row mr-4 space-x-2 justify-center items-center">
                <img
                  src={copy}
                  height={24}
                  width={24}
                  className="not-md:hidden"
                ></img>
                <img
                  src={copy}
                  height={12}
                  width={12}
                  className="md:hidden"
                ></img>
                <p className="md:text-md text-xs inter-400 text-[#797979]">
                  25
                </p>
              </div>
            </div>
            <p className="md:text-xl text-md inter-400 text-[#a1a2a2] w-[70vw] md:w-[80%] break-words">
              Virtus is a VIT based deepfake detection model built to uncover
              manipulated media-whether image or video. Fast. Accurate.
              Transparent.
            </p>
            <button className="bg-[#1E1E1E] h-[3.8rem] w-[12rem] flex flex-row justify-between items-center space-x-4 rounded-br-2xl mt-2 px-2">
              <p className="text-sm text-white inter-400">Try Veritas Now!</p>
              <div className="bg-[#606060] h-[3rem] w-[3rem] flex flex-col justify-center items-center rounded-br-2xl">
                <img src={arrow} height={12} width={12} className=""></img>
              </div>
            </button>
          </div>
        </div>
      </div>
      <Footer />
    </>
  );
}

export default Home;
