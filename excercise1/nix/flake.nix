{
  description = "Nix environment for Jupyter notebooks in Python";

  inputs = {
    nixpkgs.url = "nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
      };

      pyEnv = pkgs.python3.withPackages (p: with p; [
        jupyterlab
        matplotlib
        numpy
        scipy
      ]);

    in
    {
      devShells.default = with pkgs;
        mkShell {
          buildInputs = [ pyEnv ];
          shellHook = ''
            jupyter lab Python_intro.ipynb
          '';
        };

      formatter = pkgs.nixpkgs-fmt;
    });

}
