import NavBar from "../components/NavBar";
import background from "../assets/bg.png";
import { useGlobalContext } from "../hooks/GlobalContext";
import hack4bengal from "../assets/hack4b.png";

function Home() {
  const { isMenuOpen, changeMenu } = useGlobalContext();

  return (
    <>
      <NavBar />
      <img className="absolute h-screen w-screen z-0" src={background}></img>
      {isMenuOpen ? (
        <div className="h-screen w-screen z-20 fixed inset-0 backdrop-brightness-75"></div>
      ) : (
        <></>
      )}
      <div className="fixed -left-0 top-[40%] h-[11rem] w-[3rem] bg-white flex flex-col">
        <img src={hack4bengal} className="-rotate-90"></img>
        <p className="[writing-mode:vertical-rl] rotate-180 text-center text-xl inter-400 ml-2">
          hack4bengal
        </p>
      </div>
    </>
  );
}

export default Home;
