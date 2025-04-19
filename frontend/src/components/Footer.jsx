import { useGlobalContext } from "../hooks/GlobalContext";
import dot from "../assets/dot.svg";
import { useLocation } from "react-router";

const Footer = () => {
  const { activeTab, changeTab } = useGlobalContext();
  const location = useLocation();
  var path = location.pathname.slice(1);

  const menuItems = ["HOME", "MODELS", "PLAYGROUND", "ABOUT", "CONTACT"];

  let tab, index;
  for (let i = 0; i < menuItems.length; i++) {
    if (menuItems[i] == path.toUpperCase()) {
      tab = activeTab;
      index = i;
    }
  }

  return (
    <div className="w-screen flex justify-center items-center">
      <div className="fixed bottom-9 z-20 w-[95%] h-[1rem] border-[#1E1E1E] border-[0.08rem] border-b-0">
        <div className="fixed bottom-9 z-21 w-[35vw] h-[1rem] border-[#1E1E1E] border-[0.08rem] border-b-0 border-l-0"></div>
      </div>
      <div className="fixed bottom-6 z-20 w-[100%] h-[1rem] flex flex-row justify-between items-center text-[#797979] text-sm inter-400">
        <div className="ml-5 mt-3 flex flex-row space-x-2">
          <img src={dot} height={7} width={7}></img>
          <p>/00{index + 1}</p>
        </div>
        <div className="w-[25vw] mt-3 flex flex-row space-x-2">
          <img src={dot} height={7} width={7}></img>
          <p>({menuItems[index]})</p>
        </div>
        <p className="mt-3">Scroll Down....</p>
      </div>
    </div>
  );
};

export default Footer;
