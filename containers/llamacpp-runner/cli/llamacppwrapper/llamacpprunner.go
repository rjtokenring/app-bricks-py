package llamacppwrapper

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"syscall"
)

func generateProgressBar(currentStep int, totalSteps int, barWidth int) string {
	if totalSteps <= 0 {
		return "[]"
	}
	if currentStep < 0 {
		currentStep = 0
	}
	if currentStep > totalSteps {
		currentStep = totalSteps
	}

	progress := float64(currentStep) / float64(totalSteps)
	percent := int(progress * 100)
	filledChars := int(float64(barWidth) * progress)
	emptyChars := barWidth - filledChars

	bar := "[" + strings.Repeat("#", filledChars) + strings.Repeat("-", emptyChars) + "]"

	return fmt.Sprintf("\rProcessing: %s %3d%% Complete (%d/%d)", bar, percent, currentStep, totalSteps)
}

func DownloadMode(model string) (int, error) {
	if inf, err := os.Stat(fmt.Sprintf("%s.partial", model)); err == nil && inf.Size() > 0 {
		fmt.Println("Resuming partial download...")
	}

	cmd := exec.Command("llama-run", model, "-ngl", "16", "\"1+1=?\"")

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		panic(err)
	}
	cmd.Stderr = cmd.Stdout

	if err := cmd.Start(); err != nil {
		panic(err)
	}

	go func() {
		whitespaceRe := regexp.MustCompile(`\s+\|`)
		fullCharRe := regexp.MustCompile(`[^\x00-\x7F]+`)
		replaceTag := []byte("")

		buf := make([]byte, 2048)
		for {
			n, err := stdoutPipe.Read(buf)
			if n > 0 {
				out := whitespaceRe.ReplaceAll(buf[:n], replaceTag)
				out = fullCharRe.ReplaceAll(out, replaceTag)
				line := strings.TrimSpace(string(out))
				if line == "" {
					continue
				} else if strings.Contains(line, "%") {
					// Extract and print progress
					parts := strings.Split(line, "%")
					if len(parts) > 0 {
						progress := strings.TrimSpace(parts[0])
						if strings.Contains(progress, " ") {
							splitted := strings.Split(progress, " ")
							if len(splitted) > 1 {
								percent, err := strconv.Atoi(splitted[1])
								if err == nil {
									barOut := generateProgressBar(percent, 100, 30)
									fmt.Print(barOut)
								} else {
									fmt.Printf("\r%s%%", progress)
								}
							}
						}
					}
				}
			}
			if err != nil {
				break
			}
		}
	}()

	// Wait for command to finish
	err = cmd.Wait()

	// Get exit code
	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
				exitCode = status.ExitStatus()
			}
		} else {
			fmt.Println("Error running command:", err)
		}
	} else {
		if status, ok := cmd.ProcessState.Sys().(syscall.WaitStatus); ok {
			exitCode = status.ExitStatus()
		}
	}

	fmt.Println("Process finished with exit code:", exitCode)

	return exitCode, nil
}
