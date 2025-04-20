import { createBrowserRouter } from "react-router";
import { lazy, Suspense } from "react";
import LoadingScreen from "./utility/Loading";
import ReactLenis from "lenis/react";

const wait = (time) => {
  return new Promise((res) => {
    setTimeout(() => {
      res();
    }, time);
  });
};

const Home = lazy(() => wait(2000).then(() => import("../pages/Home")));
const Playground = lazy(() =>
  wait(2000).then(() => import("../pages/Playground"))
);
const About = lazy(() => wait(2000).then(() => import("../pages/About")));
const Contact = lazy(() => wait(2000).then(() => import("../pages/Contact")));

const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <Suspense fallback={LoadingScreen("LOADING LOPT___")}>
        <ReactLenis root>
          <Home />
        </ReactLenis>
      </Suspense>
    ),
  },
  {
    path: "/home",
    element: (
      <Suspense fallback={LoadingScreen("LOADING LOPT___")}>
        <ReactLenis root>
          <Home />
        </ReactLenis>
      </Suspense>
    ),
  },
  {
    path: "/playground",
    element: (
      <Suspense fallback={LoadingScreen("LOADING LOPT___")}>
        <ReactLenis root>
          <Playground />
        </ReactLenis>
      </Suspense>
    ),
  },
  {
    path: "/about",
    element: (
      <Suspense fallback={LoadingScreen("LOADING LOPT___")}>
        <ReactLenis root>
          <About />
        </ReactLenis>
      </Suspense>
    ),
  },
  {
    path: "/contact",
    element: (
      <Suspense fallback={LoadingScreen("LOADING LOPT___")}>
        <ReactLenis root>
          <Contact />
        </ReactLenis>
      </Suspense>
    ),
  },
]);

export default router;
