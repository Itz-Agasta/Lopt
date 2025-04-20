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
  wait(1000).then(() => import("../pages/Playground"))
);
const Models = lazy(() => wait(1000).then(() => import("../pages/Models")));

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
    path: "/models",
    element: (
      <Suspense fallback={LoadingScreen("LOADING LOPT___")}>
        <ReactLenis root>
          <Models />
        </ReactLenis>
      </Suspense>
    ),
  },
]);

export default router;
