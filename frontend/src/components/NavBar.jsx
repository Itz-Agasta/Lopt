import logo from "../assets/veritas.png";
import menu from "../assets/list.svg";
import { useEffect, useState } from "react";
import MenuBar from "../components/MenuBar";
import { useGlobalContext } from "../hooks/GlobalContext";
import { useLocation } from "react-router";

const NavBar = () => {
  const [time, setTime] = useState(new Date());
  const { activeTab, changeMenu } = useGlobalContext();
  const location = useLocation();
  var path = location.pathname.slice(1);

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
            <h1
              className={
                path === "home"
                  ? "md:text-9xl text-5xl font-[nmBold] text-white"
                  : "md:text-5xl text-5xl font-[nmBold] text-white mt-3"
              }
            >
              LOPT
            </h1>
          </div>
          <div className="text-sm inter-400 mt-5 gap-10 text-white flex flex-row justify-center">
            <ul className="not-md:hidden">
              + Where reality meets
              <br />
              AI integrity
            </ul>
            <ul className="not-md:hidden">
              + Le temps est
              <br />
              {time.toLocaleTimeString()}
            </ul>
          </div>
          <button
            className="min-w-24 mt-5 max-h-[34px] bg-[#1E1E1E] text-white inter-400 text-sm py-2 px-3 rounded-md mr-5 flex flex-row justify-center items-center hover:bg-red-500"
            onClick={changeMenu}
          >
            <img src={menu} className="mr-1"></img>MENU
          </button>
        </div>
      </div>
      <MenuBar />
    </>
  );
};

export default NavBar;
