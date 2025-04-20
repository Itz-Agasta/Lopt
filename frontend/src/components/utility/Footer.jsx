import { useGlobalContext } from "../../hooks/GlobalContext";
import dot from "../../assets/dot.svg";
import { useLocation } from "react-router";
import { useEffect, useState } from "react";

const Footer = () => {
  const { activeTab, changeTab } = useGlobalContext();
  const [index, setIndex] = useState(0);
  const location = useLocation();
  var path = location.pathname.slice(1);

  const menuItems = ["HOME", "MODELS", "PLAYGROUND", "ABOUT", "CONTACT"];
  const list = [
    "fG@3#zL!8qT$w1e",
    "^b9R&u*Xm2K!@7p",
    "C$7!vB#zNq%8e&L",
    "1x@G#M^4e!y$Z9t",
    "d&3*Q!k^wT@P#8r",
    "Scroll Down___",
    "fG@3#zL!8qT$w1e",
    "^b9R&u*Xm2K!@7p",
    "C$7!vB#zNq%8e&L",
    "1x@G#M^4e!y$Z9t",
    "d&3*Q!k^wT@P#8r",
    "Scroll Down___",
  ];

  var tab, idx;
  if (path.length > 0) {
    for (let i = 0; i < menuItems.length; i++) {
      if (menuItems[i] === path.toUpperCase()) {
        tab = activeTab;
        idx = i;
      }
    }
  } else {
    idx = 0;
  }

  useEffect(() => {
    let delay = list[index] === "Scroll Down___" ? 1000 : 200;

    const timer = setTimeout(() => {
      setIndex((prevIndex) => (prevIndex + 1) % list.length);
    }, delay);

    return () => clearTimeout(timer); // cleanup
  }, [index]);

  const [scroll, setScroll] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScroll(!scroll);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="w-screen flex justify-center items-center">
      {scroll ? (
        <div className="absolute bottom-9 z-20 w-[95%] h-[1rem] border-[#1E1E1E] border-[0.08rem] border-b-0">
          <div className="absolute bottom-9 z-21 w-[35vw] h-[1rem] border-[#1E1E1E] border-[0.08rem] border-b-0 border-l-0"></div>
        </div>
      ) : (
        <div className="fixed bottom-9 z-20 w-[95%] h-[1rem] border-[#1E1E1E] border-[0.08rem] border-b-0">
          <div className="fixed bottom-9 z-21 w-[35vw] h-[1rem] border-[#1E1E1E] border-[0.08rem] border-b-0 border-l-0"></div>
        </div>
      )}
      {scroll ? (
        <div className="absolute bottom-6 z-20 w-[100%] h-[1rem] flex flex-row justify-between items-center text-[#797979] text-xs inter-400">
          <div className="md:ml-5 mt-3 flex flex-row">
            <img src={dot} height={7} width={7}></img>
            <p>/00{idx + 1}</p>
          </div>
          <div className="w-[25vw] mt-3 flex flex-row space-x-2 not-md:absolute not-md:right-[40%]">
            <img src={dot} height={7} width={7}></img>
            <p>({menuItems[idx]})</p>
          </div>
          <div className="w-[10vw] text-right not-md:absolute not-md:right-0 not-md:w-[25vw]">
            <p className="mt-3 text-white opacity-80">{list[index]}</p>
          </div>
        </div>
      ) : (
        <div className="fixed bottom-6 z-20 w-[100%] h-[1rem] flex flex-row justify-between items-center text-[#797979] text-xs inter-400">
          <div className="md:ml-5 mt-3 flex flex-row">
            <img src={dot} height={7} width={7}></img>
            <p>/00{idx + 1}</p>
          </div>
          <div className="w-[25vw] mt-3 flex flex-row space-x-2 not-md:absolute not-md:right-[40%]">
            <img src={dot} height={7} width={7}></img>
            <p>({menuItems[idx]})</p>
          </div>
          <div className="w-[10vw] text-right not-md:absolute not-md:right-0 not-md:w-[25vw]">
            <p className="mt-3 text-white opacity-80">{list[index]}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Footer;
