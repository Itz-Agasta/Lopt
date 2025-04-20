import { NavLink, useLocation } from "react-router";
import cross from "../../assets/cross.svg";
import dot from "../../assets/dot.svg";
import github from "../../assets/github.svg";
import { useGlobalContext } from "../../hooks/GlobalContext";
import dot2 from "../../assets/dot2.svg";
import Hover from "./Hover";

const MenuBar = () => {
  const { activeTab, changeTab, isMenuOpen, changeMenu } = useGlobalContext();
  const location = useLocation();
  var path = location.pathname;
  path = path.slice(1);
  if (path.length == 0) path = "home";

  const menuItems = ["HOME", "MODELS", "PLAYGROUND", "ABOUT", "CONTACT"];

  return isMenuOpen ? (
    <>
      <div className="fixed z-200 bg-black h-[100vh] w-[40vw] -top-1 -right-0 border-[0.08rem] border-l-[#1E1E1E] flex justify-between flex-col not-md:hidden">
        <nav className="relative mt-7 ml-5 flex flex-row justify-between">
          <div className="flex flex-row items-center gap-x-2">
            <img src={dot} height={7} width={7}></img>
            <p className="text-[#797979] text-md">Menu</p>
          </div>
          <button
            className="bg-[#1E1E1E] text-white inter-400 text-sm py-2 px-3 rounded-md mr-5 min-h-[20px] min-w-[30px] flex flex-row justify-center items-center hover:scale-105"
            onClick={changeMenu}
          >
            <img src={cross} className="mr-2" height={10} width={10}></img>
            <Hover>CLOSE</Hover>
          </button>
        </nav>
        <div className="relative h-[92vh] w-[39.5vw] flex justify-center items-center">
          <div className="border-[0.08rem] border-[#1E1E1E] rounded-md flex flex-col justify-end h-[70%] w-[90%] gap-y-10 px-5">
            {menuItems.map((item, key) => (
              <ul
                className={
                  key == 4
                    ? "flex flex-row justify-between items-center gap-x-5 pb-2"
                    : "flex flex-row justify-between items-center gap-x-5 pb-2 border-b-[0.08rem] border-[#1E1E1E]"
                }
              >
                <div className="flex flex-row items-center gap-x-5">
                  <p className="text-[#797979] text-sm inter-400">
                    /0{key + 1}
                  </p>
                  <button
                    className={
                      key == 4
                        ? "text-4xl text-white font-[mubold] -mt-2"
                        : "text-4xl text-white font-[mubold]"
                    }
                    onClick={() => changeTab(item)}
                  >
                    <NavLink to={`/${item.toLowerCase()}`}>
                      <Hover>{item}</Hover>
                    </NavLink>
                  </button>
                  <img
                    src={dot2}
                    height={20}
                    width={20}
                    style={
                      path === item.toLowerCase() ? {} : { display: "none" }
                    }
                  ></img>
                </div>
                <img
                  src={dot}
                  height={10}
                  width={10}
                  style={path === item.toLowerCase() ? { display: "none" } : {}}
                ></img>
              </ul>
            ))}
          </div>
        </div>
        <button className="relative bottom-10 mx-auto  border-[0.08rem] border-[#1E1E1E] h-[4rem] w-[10rem] flex flex-row justify-center items-center space-x-3 rounded-md block">
          <img src={github} alt="Github" height={24} width={24}></img>
          <a href="https://github.com/Itz-Agasta/Lopt.git">
            <p className="text-white text-sm inter-400">Star This Project</p>
          </a>
        </button>
      </div>
      <div className="absolute z-40 bg-black h-screen w-screen top-0 md:hidden">
        <nav className="relative flex flex-row max-h-[20%] w-full justify-between mx-5">
          <div className="flex flex-row items-center gap-x-2">
            <img src={dot} height={7} width={7}></img>
            <p className="text-[#797979] text-md">Menu</p>
          </div>
          <button
            className="bg-[#1E1E1E] text-white inter-400 text-sm py-2 px-3 rounded-md mr-10 mt-5 min-h-[20px] min-w-[30px] flex flex-row justify-center items-center"
            onClick={changeMenu}
          >
            <img src={cross} className="mr-2" height={10} width={10}></img>
            CLOSE
          </button>
        </nav>
        <div className="relative h-full w-full flex justify-center items-center flex-col space-y-3">
          <div className="border-[0.08rem] border-[#1E1E1E] rounded-md flex flex-col justify-center h-[60%] w-[90%] gap-y-10 px-5">
            {menuItems.map((item, key) => (
              <ul
                className={
                  key == 4
                    ? "flex flex-row justify-between items-center gap-x-5"
                    : "flex flex-row justify-between items-center gap-x-5 pb-2 border-b-[0.08rem] border-[#1E1E1E]"
                }
              >
                <div className="flex flex-row items-center gap-x-5">
                  <p className="text-[#797979] text-sm inter-400">
                    /0{key + 1}
                  </p>
                  <button
                    className="text-3xl text-white inter-900"
                    onClick={() => changeTab(item)}
                  >
                    <NavLink to={`/${item.toLowerCase()}`}>{item}</NavLink>
                  </button>
                  <img
                    src={dot2}
                    height={20}
                    width={20}
                    style={activeTab === item ? {} : { display: "none" }}
                  ></img>
                </div>
                <img
                  src={dot}
                  height={10}
                  width={10}
                  style={activeTab === item ? { display: "none" } : {}}
                ></img>
              </ul>
            ))}
          </div>
          <button className="relative mx-auto border-[0.08rem] border-[#1E1E1E] h-[4rem] w-[10rem] flex flex-row justify-center items-center space-x-3 rounded-md block">
            {/*TODO: Enter github link*/}
            <img src={github} alt="Github" height={18} width={18}></img>
            <p className="text-white text-sm inter-400">Star This Project</p>
          </button>
        </div>
      </div>
    </>
  ) : (
    <></>
  );
};

export default MenuBar;
