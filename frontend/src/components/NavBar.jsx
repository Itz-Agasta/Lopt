import logo from "../assets/veritas.png";
import menu from "../assets/list.svg";
import { useEffect, useState } from "react";
import MenuBar from "../components/MenuBar";
import { useGlobalContext } from "../hooks/GlobalContext";

const NavBar = () => {
  const [time, setTime] = useState(new Date());
  const { changeMenu } = useGlobalContext();

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);
  return (
    <>
      <div className="absolute">
        <div className="relative z-10 flex flex-row justify-between">
          <img
            src={logo}
            className="fixed z-20 -top-10 -left-5"
            height={450}
            width={600}
          ></img>
          <div className="text-sm inter-400 mt-5 gap-10 text-white flex flex-row right-0 w-screen justify-end">
            <ul>
              + Where reality meets
              <br />
              AI integrity
            </ul>
            <ul>
              + Le temps est
              <br />
              {time.toLocaleTimeString()}
            </ul>
            <button
              className="bg-[#1E1E1E] text-white inter-400 text-sm py-2 px-3 rounded-md mr-5 min-h-[40px] min-w-[70px] flex flex-row justify-center items-center hover:bg-red-500"
              onClick={changeMenu}
            >
              <img src={menu} className="mr-1"></img>MENU
            </button>
          </div>
        </div>
        <MenuBar />
      </div>
    </>
  );
};

export default NavBar;
