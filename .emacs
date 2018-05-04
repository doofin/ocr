(defun keybd (key action) (global-set-key (kbd key) action))
(defun pkgrecompile () (byte-recompile-directory package-user-dir nil 'force))

(setq-default cursor-type '(bar . 5))
(global-linum-mode 1)
(show-paren-mode 1)
(global-auto-revert-mode t)
(tool-bar-mode -1)

(keybd "C-x l" 'goto-line)
(keybd "C-=" 'enlarge-window)
(keybd "C--" 'shrink-window)
(keybd "C-d" 'kill-whole-line)
(keybd "<M-up>" 'mark-sexp)
(keybd "<M-right>" 'sp-forward-sexp)
(keybd "<M-left>" 'sp-backward-sexp)
(keybd "C-o" 'other-window)
(keybd "C-f" 'isearch-forward)
(keybd "C-<next>" 'next-buffer)
(keybd "C-<prior>" 'previous-buffer)
(keybd "C-o" 'popup-imenu)
(global-set-key (kbd "C-p") 'insert-parentheses)
(global-set-key (kbd "M-1") 'neotree-toggle)
(global-set-key "\C-s" 'save-buffer)
(global-set-key (kbd "C-x j") 'delete-indentation)
(global-set-key "\C-v" 'yank)
(global-set-key "\C-z" 'undo)
(define-key isearch-mode-map "\C-f" 'isearch-repeat-forward)
;;(global-set-key "\C-c" 'kill-ring-save)

(setq backup-directory-alist `(("." . "~/emacs_autosave")))

(require 'package)
(setq package-archives '(
			 ("gnu" . "https://elpa.gnu.org/packages/")
                         ("melpa-stable" . "http://stable.melpa.org/packages/")))
(add-to-list
  'package-archives
  '("melpa" . "http://melpa.org/packages/") t)


(setq package-check-signature nil)
(add-to-list 'load-path "~/.emacs.d/lisp/")
(add-to-list 'load-path "~/.cabal/share/x86_64-linux-ghc-8.0.2/HaRe-0.8.4.1/elisp")
(package-initialize)

(require 'company)
(require 'rainbow-delimiters)
(add-hook 'after-init-hook 'global-company-mode)



(custom-set-faces
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(default ((t (:family "Ubuntu Mono" :foundry "unknown" :slant normal :weight normal :height 145 :width normal))))
 '(agda2-highlight-datatype-face ((t (:foreground "deep sky blue"))))
 '(agda2-highlight-function-face ((t (:foreground "deep sky blue"))))
 '(agda2-highlight-postulate-face ((t (:foreground "deep sky blue"))))
 '(agda2-highlight-primitive-face ((t (:foreground "deep sky blue"))))
 '(agda2-highlight-primitive-type-face ((t (:foreground "cyan"))))
 '(agda2-highlight-record-face ((t (:foreground "deep sky blue"))))
 '(rainbow-delimiters-depth-1-face ((t (:foreground "gainsboro"))))
 '(rainbow-delimiters-depth-2-face ((t (:foreground "magenta3"))))
 '(rainbow-delimiters-depth-3-face ((t (:foreground "medium turquoise"))))
 '(rainbow-delimiters-depth-4-face ((t (:foreground "steel blue"))))
 '(rainbow-delimiters-depth-5-face ((t (:foreground "spring green"))))
 '(rainbow-delimiters-depth-6-face ((t (:foreground "gold"))))
 '(rainbow-delimiters-depth-7-face ((t (:foreground "dark orange"))))
 '(rainbow-delimiters-depth-8-face ((t (:foreground "red"))))
 '(region ((t (:background "sea green")))))
(custom-set-variables
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(agda2-highlight-level (quote interactive))
 '(ansi-color-names-vector
   ["black" "#d55e00" "#009e73" "#f8ec59" "#0072b2" "#cc79a7" "#56b4e9" "white"])
 '(company-ghc-show-info nil)

 '(custom-enabled-themes (quote (deeper-blue)))
 '(haskell-process-auto-import-loaded-modules t)
 '(haskell-process-log t)
 '(haskell-process-suggest-remove-import-lines t)
 '(inhibit-startup-screen t))


(let ((my-cabal-path (expand-file-name "~/.cabal/bin")))
  (setenv "PATH" (concat my-cabal-path ":" (getenv "PATH")))
  (add-to-list 'exec-path my-cabal-path))
(add-to-list 'company-backends 'company-ghc)


(require 'hare)
(autoload 'hare-init "hare" nil t)

(add-hook 'haskell-mode-hook (lambda () (progn
					  (ghc-init)
					  (hare-init)
					  (keybd "C-r" 'hare-refactor-rename)
					  (define-key haskell-mode-map (kbd "C-l") 'haskell-process-load-or-reload)
					  (define-key haskell-mode-map (kbd "<f3>") 'haskell-mode-jump-to-def)
					  (keybd "<f2>" 'ghc-show-info)
					  (keybd "C-t" 'ghc-show-info)
					  (haskell-indentation-mode)
					  (interactive-haskell-mode)
					  (turn-on-haskell-indentation)
					  (global-flycheck-mode)
					  (eval-after-load 'flycheck '(require 'flycheck-hdevtools))
					  (eval-after-load 'haskell-mode '(progn
									    (haskell-session-change)
									    ))
					  (message "hs evaled!") 
					  )))





;; (load-file (let ((coding-system-for-read 'utf-8)) (shell-command-to-string "agda-mode locate")))


;(require 'helm-config)
;(helm-mode 1)

;(add-hook 'c++-mode-hook 'irony-mode)
;(add-hook 'c-mode-hook 'irony-mode)

;; replace the `completion-at-point' and `complete-symbol' bindings in
;; irony-mode's buffers by irony-mode's function
;(add-hook 'irony-mode-hook 'my-irony-mode-hook)
;(add-hook 'irony-mode-hook 'irony-cdb-autosetup-compile-options)

;(add-hook 'python-mode-hook 'jedi:setup)
;(setq jedi:complete-on-dot t)

;;(defun my/python-mode-hook ()
;;  (add-to-list 'company-backends 'company-jedi))

;;(add-hook 'python-mode-hook 'my/python-mode-hook)

(smartparens-global-mode t)
(package-initialize)

(require 'idle-highlight-mode)
(add-hook 'prog-mode-hook 'idle-highlight-mode)

;;(load "/home/d/.opam/4.04.1/share/emacs/site-lisp/tuareg-site-file")
; (load "/home/d/.opam/default/share/emacs/site-lisp/tuareg-site-file")
 ;; (let ((opam-share (ignore-errors (car (process-lines "opam" "config" "var"
 ;;   "share")))))
 ;;      (when (and opam-share (file-directory-p opam-share))
 ;;       ;; Register Merlin
 ;;       (add-to-list 'load-path (expand-file-name "emacs/site-lisp" opam-share))
 ;;       (autoload 'merlin-mode "merlin" nil t nil)
 ;;       ;; Automatically start it in OCaml buffers
 ;;       ;;(add-hook 'tuareg-mode-hook 'merlin-mode t)
 ;;       (add-hook 'tuareg-mode-hook
 ;;          '(lambda ()
 ;;             (merlin-mode t)
 ;;             (local-set-key (kbd "\C-o") 'merlin-occurrences)
 ;; 	     (local-set-key (kbd "<f3>") 'merlin-locate)
 ;; 	     (local-set-key (kbd "\C-t") 'merlin-type-enclosing)
 ;;             ))
 ;;       ;; Use opam switch to lookup ocamlmerlin binary
 ;;       (setq merlin-command 'opam)))




(use-package ensime
  :ensure t
  :pin melpa-stable)

(add-hook 'ensime-mode-hook '(lambda () (progn
					  (setq ensime-startup-notification 'nil)
					  (setq ensime-eldoc-hints 'all)
					  (git-gutter-mode)
					  (keybd "<f2>" 'ensime-type-at-point)
					  (keybd "<C-t>" 'ensime-type-at-point)
					  (keybd "<f3>" 'ensime-goto-source-location)
					  (keybd "C-x f" 'ensime-format-source)
					  (keybd "C-r" 'ensime-refactor-diff-rename)
					  (keybd "M-RET" 'ensime-refactor-add-type-annotation)
					  (message "ensime-mode customization ok!")
	     )))

(add-hook 'emacs-lisp-mode-hook '(lambda () (progn
					      (global-set-key (kbd "C-l") (lambda () (progn
										       (interactive)
										       (eval-buffer)
										       (message "elisp evaled!")
										       ))) ;; 'eval-buffer

					      )))
(message "elisp evaled!")


