import menu from "../../assets/list.svg";
import { useEffect, useState } from "react";
import MenuBar from "./MenuBar";
import Reveal1 from "./Reveal1";
import Reveal2 from "./Reveal2";
import { useGlobalContext } from "../../hooks/GlobalContext";
import { useLocation } from "react-router";
import Hover from "./Hover";
import { NavLink } from "react-router";

const NavBar = () => {
  const [scroll, setScroll] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScroll(!scroll);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const { activeTab, changeMenu } = useGlobalContext();
  const location = useLocation();
  var path = location.pathname.slice(1);
  if (path.length <= 0) path = "home";

  return (
    <>
      <div className="fixed z-200">
        <div className="relative flex flex-row justify-between w-[100vw]">
          <div className="max-w-[30vw] ml-4 not-md:mt-3">
            <Reveal1>
              {scroll ? (
                <h1
                  className={
                    path === "home"
                      ? "md:text-5xl text-3xl font-[mubold] text-white mt-3 transition-all duration-300"
                      : "md:text-5xl text-5xl font-[mubold] text-white mt-3 transition-all duration-300"
                  }
                >
                  <NavLink to="/home">LOPT</NavLink>
                </h1>
              ) : (
                <h1
                  className={
                    path === "home"
                      ? "md:text-9xl text-5xl font-[mubold] text-white transition-all duration-300"
                      : "md:text-5xl text-5xl font-[mubold] text-white mt-3 transition-all duration-300"
                  }
                >
                  <NavLink to="/home">LOPT</NavLink>
                </h1>
              )}
            </Reveal1>
          </div>
          <Reveal2>
            <button
              className="min-w-24 mt-5 max-h-[34px] bg-[#1E1E1E] text-white inter-400 text-sm py-2 px-3 rounded-md mr-5 flex flex-row justify-center items-center hover:bg-[#f03b05]"
              onClick={changeMenu}
            >
              <img src={menu} className="mr-1"></img>
              <Hover>MENU</Hover>
            </button>
          </Reveal2>
        </div>
      </div>
      <MenuBar />
    </>
  );
};

export default NavBar;
