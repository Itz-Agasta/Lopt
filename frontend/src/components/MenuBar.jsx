import { NavLink } from "react-router";
import cross from "../assets/cross.svg";
import dot from "../assets/dot.svg";
import github from "../assets/github.svg";
import { useGlobalContext } from "../hooks/GlobalContext";
import dot2 from "../assets/dot2.svg";

const MenuBar = () => {
  const { activeTab, changeTab, isMenuOpen, changeMenu } = useGlobalContext();

  const menuItems = ["HOME", "MODELS", "PLAYGROUND", "ABOUT", "CONTACT"];

  return isMenuOpen ? (
    <>
      <div className="absolute z-30 bg-black h-[100vh] w-[40vw] -top-1 -right-1 border-[0.08rem] border-l-[#1E1E1E]">
        <nav className="relative mt-7 ml-5 flex flex-row justify-between">
          <div className="flex flex-row items-center gap-x-2">
            <img src={dot} height={7} width={7}></img>
            <p className="text-[#797979] text-md">Menu</p>
          </div>

          <button
            className="bg-[#1E1E1E] text-white inter-400 text-sm py-2 px-3 rounded-md mr-5 min-h-[20px] min-w-[30px] flex flex-row justify-center items-center hover:scale-105"
            onClick={() => {
              changeMenu();
              console.log(isMenuOpen);
            }}
          >
            <img src={cross} className="mr-2" height={10} width={10}></img>
            CLOSE
          </button>
        </nav>
        <div className="relative h-[92vh] w-[39.5vw] flex justify-center items-center">
          <div className="border-[0.08rem] border-[#1E1E1E] rounded-md flex flex-col justify-center h-[70%] w-[90%] gap-y-10 px-5">
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
                    className="text-5xl text-white inter-900"
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
        </div>
        <button className="relative">
          <img src={github} height={10} width={10}></img>
        </button>
      </div>
    </>
  ) : (
    <></>
  );
};

export default MenuBar;
