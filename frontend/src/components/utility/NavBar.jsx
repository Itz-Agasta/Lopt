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
  const [time, setTime] = useState(new Date());
  const { activeTab, changeMenu } = useGlobalContext();
  const location = useLocation();
  var path = location.pathname.slice(1);
  if (path.length <= 0) path = "home";

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);
  return (
    <>
      <div className="absolute">
        <div className="relative z-10 flex flex-row justify-between w-[100vw]">
          <div className="max-w-[30vw] ml-4 not-md:mt-3">
            <Reveal1>
              <h1
                className={
                  path === "home"
                    ? "md:text-9xl text-5xl font-[mubold] text-white"
                    : "md:text-5xl text-5xl font-[mubold] text-white mt-3"
                }
              >
                <NavLink to="/home">LOPT</NavLink>
              </h1>
            </Reveal1>
          </div>
          <div className="text-sm inter-400 mt-5 gap-10 text-white flex flex-row justify-center">
            <Reveal1>
              <ul className="not-md:hidden">
                + Where reality meets
                <br />
                AI integrity
              </ul>
            </Reveal1>
            <Reveal1>
              <ul className="not-md:hidden">
                + Le temps est
                <br />
                {time.toLocaleTimeString()}
              </ul>
            </Reveal1>
          </div>
          <Reveal2>
            <button
              className="min-w-24 mt-5 max-h-[34px] bg-[#1E1E1E] text-white inter-400 text-sm py-2 px-3 rounded-md mr-5 flex flex-row justify-center items-center hover:bg-red-500"
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
