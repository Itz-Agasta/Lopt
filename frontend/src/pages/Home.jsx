import { useGlobalContext } from "../hooks/GlobalContext";
import NavBar from "../components/utility/NavBar";
import Footer from "../components/utility/Footer";
import Sticker from "../components/utility/h4b";
import background from "../assets/bg.png";
import dot from "../assets/dot3.svg";
import copy from "../assets/copy.svg";
import arrow from "../assets/arrow2.svg";
import arrow2 from "../assets/arrow3.svg";
import github from "../assets/github.svg";
import { InfiniteCarousel } from "../components/utility/Carousel";
import { NavLink, useLocation } from "react-router";
import CombinedReveal from "../components/utility/CombinedReveal";
import { motion } from "framer-motion";
import ScientificSections from "../components/utility/ScientificSections";
import Header from "../components/utility/Header";
import { useEffect } from "react";
import Lenis from "lenis";

function Home() {
  {
    /*}
  useEffect(() => {
    const lenis = new Lenis();
    function raf(time) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }
    requestAnimationFrame(raf);
  }, []);
  */
  }

  const { isMenuOpen, changeMenu } = useGlobalContext();

  return (
    <>
      <NavBar />
      <img className="absolute h-screen w-screen z-0" src={background}></img>
      <Header />
      {isMenuOpen ? (
        <div className="h-screen w-screen z-20 fixed inset-0 backdrop-brightness-75"></div>
      ) : (
        <></>
      )}
      <Sticker />
      <div className="absolute z-20 md:left-5 md:top-[70%] right-5 bottom-[15%] flex flex-col md:w-[2vw]">
        <button className="bg-[#1E1E1E] p-1">
          <a href="https://github.com/Itz-Agasta/Lopt.git">
            <img src={github} height={10} width={20} />
          </a>
        </button>
      </div>
      <div className="absolute -right-5 top-[10%] w-[100%] md:w-[55vw] justify-between flex flex-row border-l-[0.09rem] border-[#1e1e1e] pb-20 overflow-hidden" style={{ userSelect: "none" }}>
        <div className="select-none" style={{ userSelect: "none" }}>
          <CombinedReveal>
            <InfiniteCarousel />
          </CombinedReveal>
          <CombinedReveal>
            <div className="not-md:ml-2 md:w-[55vw] mt-6 mb-20">
              <div className="md:text-2xl text-lg inter-400 flex flex-row w-[90vw] md:w-[55vw] justify-between">
                <div className="flex flex-row">
                  <img src={dot}></img>
                  <span className="text-white ml-2">Truth,</span>
                  <span className="text-red-500 ml-1">Verified</span>
                </div>
                <div className="flex flex-row mr-4 space-x-2 justify-center items-center">
                  <img
                    src={copy}
                    height={28}
                    width={28}
                    className="not-md:hidden"
                  ></img>
                  <img
                    src={copy}
                    height={16}
                    width={16}
                    className="md:hidden"
                  ></img>
                  <p className="md:text-lg text-sm inter-400 text-[#797979]">
                    25
                  </p>
                </div>
              </div>
              <p className="md:text-2xl text-lg inter-400 text-[#a1a2a2] w-[70vw] md:w-[85%] break-words mt-4 leading-relaxed">
                Virtus is a VIT based deepfake detection model built to uncover
                manipulated media—whether image or video. Fast. Accurate.
                Transparent.
              </p>
              <NavLink to="/playground">
                <button className="bg-[#1E1E1E] h-[4rem] w-[14rem] flex flex-row justify-between items-center space-x-4 rounded-xs rounded-br-2xl mt-6 px-1 group">
                  <div className="flex flex-row justify-center items-center space-x-2 min-w-full h-[90%] rounded-xs rounded-br-2xl after:transition-all after:duration-400 group-hover:bg-[#f03b05] transition-colors duration-500">
                    <span
                      href="/playground"
                      className="relative inline-block w-[80%]"
                    >
                      <span className="after:content-[''] after:absolute after:left-2 after:bottom-0 after:h-[0.08rem] after:w-0 after:bg-white after:transition-all after:duration-500 group-hover:after:w-[85%] text-lg text-white inter-400">
                        Try Lopt Now!
                      </span>
                    </span>
                    <div className="bg-[#606060] h-[3.5rem] w-[3.5rem] flex flex-col justify-center items-center rounded-br-2xl ml-[0.4rem] group-hover:bg-[#cb3105] transition-colors duration-300">
                      <img
                        src={arrow}
                        height={14}
                        width={14}
                        className="absolute group-hover:hidden"
                      ></img>
                      <img
                        src={arrow2}
                        height={14}
                        width={14}
                        className="opacity-0 group-hover:opacity-100"
                      ></img>
                    </div>
                  </div>
                </button>
              </NavLink>
            </div>
          </CombinedReveal>
        </div>
      </div>
      <div className="relative top-[100vh]">
        <Footer />
        <ScientificSections />
      </div>
    </>
  );
}

export default Home;
