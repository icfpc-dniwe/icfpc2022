{
  inputs = {
      jupyterWith.url = "github:tweag/jupyterWith";
      nixpkgs.url = "github:abbradar/nixpkgs?ref=stable";
      jupyterWith.inputs.nixpkgs.follows = "nixpkgs";
      flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, jupyterWith, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          system = system;
          overlays = nixpkgs.lib.attrValues jupyterWith.overlays;
        };
        iPython = pkgs.kernels.iPythonWith {
          name = "Python-env";
          packages = p: with p; [
            numpy
            numba
            pillow
            opencv4
            scikitlearn
            scikitimage
	  ];
          ignoreCollisions = true;
        };
        jupyterEnvironment = pkgs.jupyterlabWith {
          kernels = [ iPython ];
        };
      in rec {
        apps.jupyterlab = {
          type = "app";
          program = "${jupyterEnvironment}/bin/jupyter-lab";
        };
        defaultApp = apps.jupyterlab;
        devShell = jupyterEnvironment.env;
      }
    );
}
