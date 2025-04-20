import { createBrowserRouter } from "react-router";
import { lazy } from "react";

const wait = (time) => {
  return new Promise((res) => {
    setTimeout(() => {
      res();
    }, time);
  });
};

const Home = lazy(() => wait(1000).then(() => import("../pages/Home")));
const Playground = lazy(() =>
  wait(1000).then(() => import("../pages/Playground"))
);
const Models = lazy(() => wait(1000).then(() => import("../pages/Models")));

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/home",
    element: <Home />,
  },
  {
    path: "/playground",
    element: <Playground />,
  },
  {
    path: "/models",
    element: <Models />,
  },
]);

export default router;
