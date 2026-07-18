{
  description = "MAMUT-routing-tools dev shell (extends nix-dev-base)";

  inputs = {
    base.url = "path:/home/onyr/nix-dev-base";
    nixpkgs.follows = "base/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { base, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      in {
        devShells.default = pkgs.mkShell {
          inputsFrom = [ base.devShells.${system}.default ];
          packages = with pkgs; [ ];
          shellHook = ''
            echo "[MAMUT-routing-tools] dev shell active (extends nix-dev-base)."
          '';
        };
      });
}
