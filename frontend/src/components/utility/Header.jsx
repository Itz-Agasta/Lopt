import { useEffect, useState } from "react";
import Reveal1 from "./Reveal1";

const Header = () => {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="absolute left-[40%] text-sm inter-400 mt-5 gap-10 text-white flex flex-row justify-center">
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
  );
};

export default Header;
