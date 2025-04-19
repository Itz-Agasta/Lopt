import { Component } from "react";
import { createBrowserRouter } from "react-router";
import Home from "../pages/Home";
import Playground from "../pages/Playground";
import Results from "../pages/Results";

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
    path: "/results",
    element: <Results />,
  },
]);

export default router;
