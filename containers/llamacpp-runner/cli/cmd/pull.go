package cmd

import (
	"fmt"

	"github.com/arduino/app-bricks-py/model-downloader/llamacppwrapper"
	"github.com/spf13/cobra"
)

// pullCmd represents the pull command
var pullCmd = &cobra.Command{
	Use:   "pull",
	Short: "Pull a model from a remote repository",
	Long:  `Pull a model from a remote repository. Supported sources: Ollama library.`,
	Run: func(cmd *cobra.Command, args []string) {
		downloadModel(cmd)
	},
}

func init() {
	rootCmd.AddCommand(pullCmd)
	pullCmd.Flags().StringP("model", "m", "", "Model to pull. E.g.: gemma3:1b")
}

func downloadModel(cmd *cobra.Command) error {
	if model, err := cmd.Flags().GetString("model"); err != nil {
		return err
	} else {
		if model == "" {
			return fmt.Errorf("model flag is required")
		}

		fmt.Printf("Pulling model: %s\n", model)
		exitCode, err := llamacppwrapper.DownloadMode(model)
		if err != nil {
			return err
		}
		if exitCode != 0 {
			return fmt.Errorf("failed to pull model, exit code: %d", exitCode)
		} else {
			fmt.Println("Model pulled successfully")
		}
	}

	return nil
}
